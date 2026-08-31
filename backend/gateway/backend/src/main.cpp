// ====================== gateway/main.cpp (modified) ======================
#include <grpcpp/grpcpp.h>
#include <gateway.grpc.pb.h>

#include <chrono>
#include <thread>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <csignal>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>

#include <curl/curl.h>
#include "third_party_json.hpp"
using json = nlohmann::json;

// === NEW: HMAC / OpenSSL ===
#include <openssl/hmac.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using gateway::v1::GatewayService;
using gateway::v1::SearchRequest;
using gateway::v1::SearchResponse;
using gateway::v1::PackShape;

// ---- forward declaration for parallel candidate fetching ----
static bool call_index_A_parallel(const std::vector<std::string>&, uint32_t, json&, int&, long&, std::string&);

// -------------------- 配置与环境 --------------------
static std::string g_index_base = "http://127.0.0.1:18081";
static std::string g_he_base    = "http://127.0.0.1:18083";
static std::string g_cfg_file;
static int ENV_DIAG_DIM() { const char* v = getenv("DIAG_DIM"); return v ? std::stoi(v) : 768; }

static int  ENV_L()         { const char* v=getenv("GATEWAY_L");     return v?std::stoi(v):8; }
static int  ENV_R()         { const char* v=getenv("GATEWAY_R");     return v?std::stoi(v):512; }
static int  ENV_CTBYTES()   { const char* v=getenv("CT_BYTES");      return v?std::stoi(v):16384; }
static int  ENV_SLOTS()     { const char* v=getenv("PACK_SLOTS");    return v?std::stoi(v):4096; }
static int  ENV_BSGS_B()    { const char* v=getenv("BSGS_B");       return v?std::stoi(v):32; }
static int  ENV_P95MS()     { const char* v=getenv("P95_MS");        return v?std::stoi(v):120; }
static int  ENV_JITTER()    { const char* v=getenv("JITTER_MS");     return v?std::stoi(v):15; }
static int  ENV_OUTCOUNT()  { const char* v=getenv("OUT_CT_COUNT");  return v?std::stoi(v):8; }

// HTTP 重试控制
static int  ENV_RETRY_N()   { const char* v=getenv("HTTP_RETRY_N");  return v?std::stoi(v):2; }
static int  ENV_RETRY_MS()  { const char* v=getenv("HTTP_RETRY_MS"); return v?std::stoi(v):120; }
static int  ENV_RJITTER()   { const char* v=getenv("RETRY_JITTER");  return v?std::stoi(v):30; }
static long ENV_HTTP_TIMEOUT_SEC() { const char* v=getenv("HTTP_TIMEOUT_SEC"); return v?std::stol(v):300L; }

// 限流
static double ENV_G_QPS(){ const char* v=getenv("RATE_GLOBAL_QPS"); return v?std::stod(v):50.0; }
static double ENV_S_QPS(){ const char* v=getenv("RATE_SESSION_QPS");return v?std::stod(v):5.0; }
static double ENV_BURST(){ const char* v=getenv("RATE_BURST");      return v?std::stod(v):2.0; }

// 配额（按会话，每小时）
static int ENV_Q_SESSION_H(){ const char* v=getenv("QUOTA_SESSION_PER_H"); return v?std::stoi(v):1000; }

// 熔断
static int ENV_CB_FAILS(){ const char* v=getenv("CB_FAILS"); return v?std::stoi(v):5; }
static int ENV_CB_OPEN_MS(){ const char* v=getenv("CB_OPEN_MS"); return v?std::stoi(v):2000; }

// A并发
static int ENV_A_PARALLEL() { const char* v=getenv("A_PARALLEL"); return v?std::stoi(v):1; }

// 伪查询策略
static std::vector<double> g_center_weights;
static std::string ENV_FAKE_POLICY(){ const char* v=getenv("FAKE_POLICY"); return v?std::string(v):"uniform"; }

// -------------------- 配置加载 / 验证 / 热更 --------------------
static void load_config(const char* path) {
  try {
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "[Gateway] config not found: " << path << "\n"; return; }
    json cfg = json::parse(ifs);

    if (cfg.contains("services")) {
      auto s = cfg["services"];
      if (s.contains("index_base")) g_index_base = s["index_base"].get<std::string>();
      if (s.contains("he_base"))    g_he_base    = s["he_base"].get<std::string>();
    }
    if (cfg.contains("gateway")) {
      auto g = cfg["gateway"];
      auto seti = [&](const char* env, const char* k){
        if (g.contains(k)) setenv(env, std::to_string(g[k].get<int>()).c_str(), 1);
      };
      if (g.contains("listen")) setenv("GATEWAY_LISTEN", g["listen"].get<std::string>().c_str(), 1);
      seti("GATEWAY_L","L"); seti("GATEWAY_R","R"); seti("CT_BYTES","ct_bytes");
      seti("PACK_SLOTS","pack_slots"); seti("P95_MS","p95_ms"); seti("JITTER_MS","jitter_ms");
      seti("OUT_CT_COUNT","out_ct_count"); seti("HTTP_RETRY_N","http_retry_n");
      seti("HTTP_RETRY_MS","http_retry_ms"); seti("RETRY_JITTER","http_retry_jitter");
      if (g.contains("rate_global_qps")) setenv("RATE_GLOBAL_QPS", std::to_string(g["rate_global_qps"].get<int>()).c_str(), 1);
      if (g.contains("rate_session_qps"))setenv("RATE_SESSION_QPS",std::to_string(g["rate_session_qps"].get<int>()).c_str(), 1);
      if (g.contains("rate_burst"))      setenv("RATE_BURST",      std::to_string(g["rate_burst"].get<int>()).c_str(), 1);

      if (g.contains("fake_policy")) setenv("FAKE_POLICY", g["fake_policy"].get<std::string>().c_str(), 1);
      if (g.contains("center_weights") && g["center_weights"].is_array()) {
        g_center_weights.clear(); for (auto& w : g["center_weights"]) g_center_weights.push_back(w.get<double>());
      }
      if (g.contains("quota_session_per_h")) setenv("QUOTA_SESSION_PER_H", std::to_string(g["quota_session_per_h"].get<int>()).c_str(), 1);

      if (g.contains("cb_fails")) setenv("CB_FAILS", std::to_string(g["cb_fails"].get<int>()).c_str(), 1);
      if (g.contains("cb_open_ms")) setenv("CB_OPEN_MS", std::to_string(g["cb_open_ms"].get<int>()).c_str(), 1);
    }
    std::cerr << "[Gateway] loaded config: " << path << "\n";
  } catch (const std::exception& e) {
    std::cerr << "[Gateway] load_config error: " << e.what() << "\n";
  }
}

static void validate_config() {
  auto bound_i = [](int v, int lo, int hi, const char* k, int defv){
    if (v<lo || v>hi){ std::cerr<<"[Config] Invalid "<<k<<": "<<v<<", reset to "<<defv<<"\n"; return defv; }
    return v;
  };
  auto bound_f = [](double v, double lo, double hi, const char* k, double defv){
    if (!(v==v) || v<lo || v>hi){ std::cerr<<"[Config] Invalid "<<k<<": "<<v<<", reset to "<<defv<<"\n"; return defv; }
    return v;
  };

  setenv("GATEWAY_L",     std::to_string(bound_i(ENV_L(),1,1000,"GATEWAY_L",8)).c_str(),1);
  setenv("GATEWAY_R",     std::to_string(bound_i(ENV_R(),1,10000,"GATEWAY_R",512)).c_str(),1);
  setenv("CT_BYTES",      std::to_string(bound_i(ENV_CTBYTES(),1,1000000,"CT_BYTES",16384)).c_str(),1);
  setenv("PACK_SLOTS",    std::to_string(bound_i(ENV_SLOTS(),1,100000,"PACK_SLOTS",4096)).c_str(),1);
  setenv("P95_MS",        std::to_string(bound_i(ENV_P95MS(),0,60000,"P95_MS",120)).c_str(),1);
  setenv("JITTER_MS",     std::to_string(bound_i(ENV_JITTER(),0,10000,"JITTER_MS",15)).c_str(),1);
  setenv("OUT_CT_COUNT",  std::to_string(bound_i(ENV_OUTCOUNT(),1,1000,"OUT_CT_COUNT",8)).c_str(),1);
  setenv("HTTP_RETRY_N",  std::to_string(bound_i(ENV_RETRY_N(),0,10,"HTTP_RETRY_N",2)).c_str(),1);
  setenv("HTTP_RETRY_MS", std::to_string(bound_i(ENV_RETRY_MS(),1,60000,"HTTP_RETRY_MS",120)).c_str(),1);
  setenv("RETRY_JITTER",  std::to_string(bound_i(ENV_RJITTER(),0,60000,"RETRY_JITTER",30)).c_str(),1);
  setenv("RATE_GLOBAL_QPS", std::to_string(bound_f(ENV_G_QPS(),0.001,100000.0,"RATE_GLOBAL_QPS",50.0)).c_str(),1);
  setenv("RATE_SESSION_QPS",std::to_string(bound_f(ENV_S_QPS(),0.001,10000.0,"RATE_SESSION_QPS",5.0)).c_str(),1);
  setenv("RATE_BURST",     std::to_string(bound_f(ENV_BURST(),0.1,1000.0,"RATE_BURST",2.0)).c_str(),1);
  setenv("QUOTA_SESSION_PER_H", std::to_string(bound_i(ENV_Q_SESSION_H(),1,1000000,"QUOTA_SESSION_PER_H",1000)).c_str(),1);
  setenv("CB_FAILS",     std::to_string(bound_i(ENV_CB_FAILS(),1,100,"CB_FAILS",5)).c_str(),1);
  setenv("CB_OPEN_MS",   std::to_string(bound_i(ENV_CB_OPEN_MS(),1,60000,"CB_OPEN_MS",2000)).c_str(),1);

  std::string policy = ENV_FAKE_POLICY();
  if (policy!="uniform" && policy!="dist_sim"){ std::cerr<<"[Config] Invalid FAKE_POLICY: "<<policy<<", reset to uniform\n"; setenv("FAKE_POLICY","uniform",1); }

  if (g_index_base.empty() || g_index_base.find("http")!=0){ std::cerr<<"[Config] Invalid index_base, reset\n"; g_index_base="http://127.0.0.1:18081"; }
  if (g_he_base.empty() || g_he_base.find("http")!=0){ std::cerr<<"[Config] Invalid he_base, reset\n"; g_he_base="http://127.0.0.1:18082"; }

  std::cerr << "[Config] Validation completed\n";
}

static void hup_handler(int){
  if (!g_cfg_file.empty()) {
    std::cerr << "[Gateway] SIGHUP: reload config\n";
    load_config(g_cfg_file.c_str());
    validate_config();
  }
}

// -------------------- Base64 编解码 --------------------
static std::string b64decode(const std::string& in) {
  static int8_t T[256]; static bool inited = false;
  if (!inited) {
    for (int i=0;i<256;i++) T[i] = -1;
    T['\t']=T['\n']=T['\r']=T[' ']=-2;
    for (int c='A'; c<='Z'; ++c) T[c] = c - 'A';
    for (int c='a'; c<='z'; ++c) T[c] = c - 'a' + 26;
    for (int c='0'; c<='9'; ++c) T[c] = c - '0' + 52;
    T['+'] = 62; T['/'] = 63; T['='] = 0;
    inited = true;
  }
  std::string out; out.reserve(in.size()*3/4 + 3);
  int val=0, valb=-8;
  for (unsigned char c : in) {
    int8_t d = T[c];
    if (d == -2) continue;
    if (d == -1) { if (c=='=') break; else continue; }
    val = (val<<6) + d; valb += 6;
    if (valb >= 0) { out.push_back(char((val>>valb)&0xFF)); valb -= 8; }
  }
  return out;
}

static std::string b64encode(const std::string& bytes) {
  static const char* T =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out; out.reserve((bytes.size() + 2)/3*4);
  int val = 0, valb = -6;
  for (unsigned char c : bytes) {
    val = (val << 8) + c; valb += 8;
    while (valb >= 0) { out.push_back(T[(val >> valb) & 0x3F]); valb -= 6; }
  }
  if (valb > -6) out.push_back(T[((val << 8) >> (valb + 8)) & 0x3F]);
  while (out.size() % 4) out.push_back('=');
  return out;
}

// -------------------- HTTP helpers + 重试 --------------------
static size_t wrfunc(void* in, size_t sz, size_t nm, void* out) {
  auto* s = reinterpret_cast<std::string*>(out);
  size_t total = sz * nm; s->append(reinterpret_cast<char*>(in), total); return total;
}

static bool http_get_json(const std::string& url, json& out) {
  CURL* c = curl_easy_init(); if (!c) return false;
  std::string buf;
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_TIMEOUT, 5L);
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, wrfunc);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &buf);
  auto rc = curl_easy_perform(c);
  long code = 0; curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
  curl_easy_cleanup(c);
  if (rc != CURLE_OK || code != 200) return false;
  try { out = json::parse(buf); return true; } catch (...) { return false; }
}

static bool http_post_json(const std::string& url, const json& body, json& out) {
  CURL* c = curl_easy_init(); if (!c) return false;
  std::string buf; auto s = body.dump();
  struct curl_slist* headers = nullptr;
  headers = curl_slist_append(headers, "Content-Type: application/json");
  curl_easy_setopt(c, CURLOPT_URL, url.c_str());
  curl_easy_setopt(c, CURLOPT_TIMEOUT, ENV_HTTP_TIMEOUT_SEC());
  curl_easy_setopt(c, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(c, CURLOPT_POST, 1L);
  curl_easy_setopt(c, CURLOPT_POSTFIELDS, s.c_str());
  curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, wrfunc);
  curl_easy_setopt(c, CURLOPT_WRITEDATA, &buf);
  auto rc = curl_easy_perform(c);
  long code = 0; curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
  curl_slist_free_all(headers); curl_easy_cleanup(c);
  if (rc != CURLE_OK || code != 200) return false;
  try { out = json::parse(buf); return true; } catch (...) { return false; }
}

static void sleep_ms(int ms){ if(ms>0) std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
static int backoff_ms(int attempt, int base, int jitter){
  long ms = long(base) * (1L<<attempt);
  if (jitter>0) ms += rand() % jitter;
  return (int)ms;
}

static bool http_get_json_retry(const std::string& url, json& out, int& retries, long& ms_spent){
  int N = ENV_RETRY_N(), base = ENV_RETRY_MS(), jit = ENV_RJITTER(); auto t0=std::chrono::steady_clock::now();
  for(int i=0;i<=N;i++){
    if (http_get_json(url, out)) { retries=i; ms_spent=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count(); return true; }
    if (i<N){ int ms=backoff_ms(i,base,jit); std::cerr<<"[Gateway] GET retry "<<(i+1)<<" in "<<ms<<"ms "<<url<<"\n"; sleep_ms(ms); }
  }
  retries=N+1; ms_spent=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count(); return false;
}

static bool http_post_json_retry(const std::string& url, const json& body, json& out, int& retries, long& ms_spent){
  int N = ENV_RETRY_N(), base = ENV_RETRY_MS(), jit = ENV_RJITTER(); auto t0=std::chrono::steady_clock::now();
  for(int i=0;i<=N;i++){
    if (http_post_json(url, body, out)) { retries=i; ms_spent=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count(); return true; }
    if (i<N){ int ms=backoff_ms(i,base,jit); std::cerr<<"[Gateway] POST retry "<<(i+1)<<" in "<<ms<<"ms "<<url<<"\n"; sleep_ms(ms); }
  }
  retries=N+1; ms_spent=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-t0).count(); return false;
}

// -------------------- 限流（全局+会话） --------------------
struct TokenBucket {
  double tokens{1}, rate{1}, burst{1};
  std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
  bool allow() {
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last).count();
    last = now; tokens = std::min(burst, tokens + dt * rate);
    if (tokens >= 1.0) { tokens -= 1.0; return true; }
    return false;
  }
};
static TokenBucket g_bucket;
static std::unordered_map<std::string, TokenBucket> g_sbuckets;
static std::mutex g_mu;
static bool rate_allow(const std::string& sid, double gr, double sr, double burst) {
  std::lock_guard<std::mutex> lk(g_mu);
  g_bucket.rate = gr; g_bucket.burst = burst;
  auto& tb = g_sbuckets[sid]; tb.rate = sr; tb.burst = burst;
  return g_bucket.allow() && tb.allow();
}

// -------------------- 会话配额（每小时） --------------------
struct Quota { int used{0}; std::chrono::steady_clock::time_point win_start{std::chrono::steady_clock::now()}; };
static std::unordered_map<std::string, Quota> g_quota_map;
static std::mutex g_quota_mutex;

static bool quota_check(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(g_quota_mutex);
  auto& q = g_quota_map[session_id];
  auto now = std::chrono::steady_clock::now();
  if (std::chrono::duration_cast<std::chrono::hours>(now - q.win_start).count() >= 1) {
    q.used = 0; q.win_start = now;
  }
  if (q.used >= ENV_Q_SESSION_H()) return false;
  q.used++; return true;
}

// -------------------- 熔断器 --------------------
struct CircuitBreaker {
  std::atomic<int> fail_count{0};
  std::atomic<bool> is_open{false};
  std::chrono::steady_clock::time_point open_until;
  std::mutex state_mutex;
};
static CircuitBreaker g_cb;

static bool circuit_breaker_allow() {
  if (!g_cb.is_open.load()) return true;
  auto now = std::chrono::steady_clock::now();
  if (now >= g_cb.open_until) {
    std::lock_guard<std::mutex> lk(g_cb.state_mutex);
    if (now >= g_cb.open_until) {
      g_cb.is_open = false; g_cb.fail_count = 0;
      std::cerr << "[CircuitBreaker] half-open\n";
      return true;
    }
  }
  return false;
}
static void circuit_breaker_on_success(){ std::lock_guard<std::mutex> lk(g_cb.state_mutex); g_cb.fail_count=0; g_cb.is_open=false; }
static void circuit_breaker_on_failure(){
  std::lock_guard<std::mutex> lk(g_cb.state_mutex);
  if (++g_cb.fail_count >= ENV_CB_FAILS() && !g_cb.is_open) {
    g_cb.is_open = true;
    g_cb.open_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(ENV_CB_OPEN_MS());
    std::cerr << "[CircuitBreaker] OPEN "<<ENV_CB_OPEN_MS()<<"ms\n";
  }
}

// -------------------- A：候选 + 对角块（公共工具） --------------------
static inline bool is_digit_only(const std::string& s) {
  if (s.empty()) return false;
  for (unsigned char c : s) if (c < '0' || c > '9') return false;
  return true;
}

// -------------------- NEW: HMAC 工具与 Nonce 缓存 --------------------
static std::vector<unsigned char> hex2bytes(const std::string& hex) {
  std::vector<unsigned char> out;
  if (hex.size() % 2) return out;
  out.reserve(hex.size()/2);
  auto val = [](char c)->int{
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return 10+(c-'a');
    if (c>='A'&&c<='F') return 10+(c-'A');
    return -1;
  };
  for (size_t i=0;i<hex.size();i+=2) {
    int hi=val(hex[i]), lo=val(hex[i+1]);
    if (hi<0||lo<0) { out.clear(); return out; }
    out.push_back(static_cast<unsigned char>((hi<<4)|lo));
  }
  return out;
}

static std::string hmac_sha256_hex(const std::vector<unsigned char>& key,
                                   const unsigned char* data, size_t len) {
  unsigned char md[EVP_MAX_MD_SIZE]; unsigned int mdlen=0;
  HMAC(EVP_sha256(), key.data(), (int)key.size(), data, len, md, &mdlen);
  static const char* H = "0123456789abcdef";
  std::string hex; hex.resize(mdlen*2);
  for (unsigned int i=0;i<mdlen;i++) {
    hex[i*2]=H[(md[i]>>4)&0xF]; hex[i*2+1]=H[md[i]&0xF];
  }
  return hex;
}

static std::vector<unsigned char> g_hmac_key; // 共享密钥
static std::once_flag g_hmac_once;

static void load_hmac_key_once() {
  std::string key_hex;
  const char* env = getenv("HMAC_KEY_HEX");
  if (env && *env) key_hex = env;
  if (key_hex.empty()) {
    std::ifstream ifs(std::string(getenv("HOME")?getenv("HOME"):"") + "/.keys/hmac_key.hex");
    if (ifs) { std::getline(ifs, key_hex); }
  }
  if (!key_hex.empty()) {
    auto k = hex2bytes(key_hex);
    if (!k.empty()) g_hmac_key = std::move(k);
  }
  if (g_hmac_key.empty()) {
    std::cerr << "[Gateway][WARN] HMAC key not configured; requests will be accepted without MAC verification.\n";
  } else {
    std::cerr << "[Gateway] HMAC key loaded ("<< g_hmac_key.size() <<" bytes)\n";
  }
}

// 简单的 nonce 去重缓存（内存，TTL=300s）
struct NonceInfo { std::chrono::steady_clock::time_point t; };
static std::unordered_map<std::string, NonceInfo> g_nonce_map;
static std::mutex g_nonce_mu;

static bool nonce_seen_or_insert(const std::string& nonce, int ttl_sec=300) {
  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lk(g_nonce_mu);
  // 清理过期
  for (auto it=g_nonce_map.begin(); it!=g_nonce_map.end();) {
    if (std::chrono::duration_cast<std::chrono::seconds>(now - it->second.t).count() > ttl_sec) it = g_nonce_map.erase(it);
    else ++it;
  }
  auto it = g_nonce_map.find(nonce);
  if (it != g_nonce_map.end()) return true; // seen
  g_nonce_map[nonce] = {now};
  return false;
}

static std::string join_clusters(const std::vector<std::string>& v) {
  std::string s; s.reserve(v.size()*4);
  for (size_t i=0;i<v.size();++i){ if(i) s.push_back(','); s += v[i]; }
  return s;
}

static bool meta_get(const std::multimap<grpc::string_ref, grpc::string_ref>& md,
                     const char* key, std::string& out) {
  auto range = md.equal_range(key);
  if (range.first == range.second) return false;
  out.assign(range.first->second.data(), range.first->second.size());
  return true;
}

// -------------------- A：候选 + 对角块（串行） --------------------
static bool call_index_A_serial(const std::vector<std::string>& clusters, uint32_t R,
                                json& out_diag_blocks, int& http_retries_a,
                                long& http_ms_a, std::string& err_tag) {
  out_diag_blocks = json::object(); // 期望 {"blocks":[...]}
  bool ok = true; http_retries_a = 0; http_ms_a = 0;

  // 逐簇取 candidates，按簇顺序拼接；假簇→R个0；真簇不足R补0
  json all_candidates = json::array();

  for (const auto& cid : clusters) {
    if (!is_digit_only(cid)) {
      for (uint32_t i = 0; i < R; ++i) all_candidates.push_back((uint64_t)0);
      continue;
    }

    json j; int r = 0; long ms = 0;
    std::string url = g_index_base + "/clusters/" + cid + "/candidates?top=" + std::to_string(R);
    if (!http_get_json_retry(url, j, r, ms)) {
      ok = false; err_tag = "A_candidates"; j = json::object();
    }
    http_retries_a += r; http_ms_a += ms;

    std::vector<uint64_t> ids;
    if (j.contains("candidates") && j["candidates"].is_array()) {
      for (auto& v : j["candidates"]) {
        if (v.is_object() && v.contains("id")) {
          ids.push_back(v["id"].get<uint64_t>());
        } else if (v.is_number_unsigned()) {
          ids.push_back(v.get<uint64_t>());
        } else if (v.is_number_integer()) {
          auto tmp = v.get<long long>(); ids.push_back(tmp < 0 ? 0ULL : (uint64_t)tmp);
        }
      }
    }
    while (ids.size() < R) ids.push_back(0ULL);
    if (ids.size() > R) ids.resize(R);
    for (auto id : ids) all_candidates.push_back(id);
  }

  json req_body = {
    {"candidates", all_candidates},       // 数字数组
    {"pack_slots", ENV_SLOTS()},
    {"dim", ENV_DIAG_DIM()}
  };

  json jr; int r = 0; long ms = 0;
  if (!http_post_json_retry(g_index_base + "/diag-blocks", req_body, jr, r, ms)) {
    ok = false; err_tag = err_tag.empty() ? "A_diag_blocks" : err_tag;
  } else {
    // 保存 /diag-blocks 返回
    try {
      std::ofstream ofs("/tmp/diag_blocks.last.json", std::ios::binary);
      ofs << jr.dump(2);
    } catch (...) { /* ignore */ }

    if (jr.contains("blocks")) out_diag_blocks = jr;
    else { ok = false; err_tag = "A_diag_blocks_shape"; }
  }
  http_retries_a += r; http_ms_a += ms;
  return ok;
}


// -------------------- B：同态打分 + 恒定化打包 --------------------
static bool call_he_B_and_pack(const SearchRequest& req, SearchResponse* resp,
                               const std::vector<std::string>& /*clusters*/, const json& diag_blocks,
                               int& http_retries_b, long& http_ms_b, std::string& err_tag, const std::string& query_mode) {
  const int out_n  = ENV_OUTCOUNT();
  int ct_sz  = ENV_CTBYTES();                // 将在需要时放大
  const uint32_t slots = (uint32_t)ENV_SLOTS();

  // 1) 查询密文 → b64
  const std::string ct_bytes(req.ct_q().data(), req.ct_q().size());
  const std::string ct_b64 = b64encode(ct_bytes);
  std::cerr << "[Gateway] B request ct_q bytes=" << ct_bytes.size() << "\n";

  // 2) 基于 diag_blocks 推导 max_off
  uint32_t max_off = 0;
  if (diag_blocks.contains("blocks") && diag_blocks["blocks"].is_array()) {
    for (auto& blk : diag_blocks["blocks"]) {
      if (blk.contains("layout") && blk["layout"].is_object() &&
          blk["layout"].contains("diag_offsets") &&
          blk["layout"]["diag_offsets"].is_array()) {
        for (auto& v : blk["layout"]["diag_offsets"]) {
          if (v.is_number_unsigned()) max_off = std::max<uint32_t>(max_off, v.get<uint32_t>());
          else if (v.is_number_integer()) {
            auto t = v.get<long long>(); if (t > 0) max_off = std::max<uint32_t>(max_off, (uint32_t)t);
          }
        }
      } else if (blk.contains("diag_offsets") && blk["diag_offsets"].is_array()) {
        for (auto& v : blk["diag_offsets"]) {
          if (v.is_number_unsigned()) max_off = std::max<uint32_t>(max_off, v.get<uint32_t>());
          else if (v.is_number_integer()) {
            auto t = v.get<long long>(); if (t > 0) max_off = std::max<uint32_t>(max_off, (uint32_t)t);
          }
        }
      } else if (blk.contains("diag_cnt") && blk["diag_cnt"].is_number_unsigned()) {
        uint32_t cnt = blk["diag_cnt"].get<uint32_t>();
        if (cnt > 0) max_off = std::max<uint32_t>(max_off, cnt - 1);
      }
    }
  }
  if (max_off == 0) {
    int D = ENV_DIAG_DIM(); if (D <= 0) D = 512;
    max_off = (uint32_t)(D - 1);
  }

  // 3) BSGS：默认 b=32；必须与离线预旋转对角块使用的 baby size 一致。
  const uint32_t b = (uint32_t)std::max(1, ENV_BSGS_B());
  const uint32_t g = (max_off + 1 + b - 1) / b;
  std::vector<uint32_t> baby(b), giant(g);
  for (uint32_t i = 0; i < b; ++i) baby[i] = i;
  for (uint32_t i = 0; i < g; ++i) giant[i] = i * b;

  // 4) 组装 HTTP 请求
  json jreq = {
    {"client_id","gw"},
    {"key_ver",  req.key_ver()},
    {"ct_q",     ct_b64},
    {"blocks",   diag_blocks.value("blocks", json::array())},
    {"diag_blocks", diag_blocks},
    {"bsgs",       {{"giant", json::array()}, {"baby", json::array()}}},
    {"bsgs_plan",  {{"giant", json::array()}, {"baby", json::array()}}},
    {"scale",    req.scale()},
    {"mod_chain", json::array({"Q0","Q1","Q2"})}
  };
  jreq["mode"] = query_mode;
  for (auto v : baby)  jreq["bsgs"]["baby"].push_back(v);
  for (auto v : giant) jreq["bsgs"]["giant"].push_back(v);
  jreq["bsgs_plan"]["baby"]  = jreq["bsgs"]["baby"];
  jreq["bsgs_plan"]["giant"] = jreq["bsgs"]["giant"];

  // 5) 调用 he_http_adapter
  json jr; int r=0; long ms=0;
  bool ok = http_post_json_retry(g_he_base + "/score/batch", jreq, jr, r, ms);
  http_retries_b = r; http_ms_b = ms;

  // 6) 取回密文数组
  auto take_scores = [&](const json& J)->const json*{
    if (J.contains("packed_scores") && J["packed_scores"].is_array()) return &J["packed_scores"];
    if (J.contains("scoresCiphertexts") && J["scoresCiphertexts"].is_array()) return &J["scoresCiphertexts"];
    return nullptr;
  };

  int got=0;
  if (ok) {
    if (auto arr = take_scores(jr)) {
      if (ct_sz < 200000 && !arr->empty()) {
        std::string dec0 = b64decode((*arr)[0].get<std::string>());
        if ((int)dec0.size() > ct_sz) {
          std::cerr << "[Gateway][WARN] CT_BYTES=" << ct_sz
                    << " too small; using real size=" << dec0.size() << "\n";
          ct_sz = (int)dec0.size();
        }
      }
      for (auto& v : *arr) {
        std::string dec = b64decode(v.get<std::string>());
        if ((int)dec.size() < ct_sz) dec.append(ct_sz - (int)dec.size(), '\0');
        // 建议不要截断密文，如确需恒定化裁剪再打开下一行
        // else if ((int)dec.size() > ct_sz) dec.resize(ct_sz);
        resp->add_scores_ciphertexts(dec);
        auto* s = resp->add_pack_shapes(); s->set_batch(1); s->set_slots( (uint32_t)ENV_SLOTS() );
        if (++got == ENV_OUTCOUNT()) break;
      }
    } else {
      err_tag = err_tag.empty()?"B_score":err_tag;
    }
  } else {
    err_tag = err_tag.empty()?"B_score_http":err_tag;
  }

  for (; got < ENV_OUTCOUNT(); ++got) {
    resp->add_scores_ciphertexts(std::string(ct_sz, '\0'));
    auto* s = resp->add_pack_shapes(); s->set_batch(1); s->set_slots( (uint32_t)ENV_SLOTS() );
  }
  return ok;
}

// -------------------- 健康探针与 eval-keys 代理（HTTP:127.0.0.1:8080） --------------------
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

static void http_send(int fd, int code, const std::string& body){
  char head[256];
  int n = snprintf(head, sizeof(head),
    "HTTP/1.1 %d %s\r\n"
    "Content-Type: application/json\r\n"
    "Content-Length: %zu\r\n"
    "Connection: close\r\n\r\n",
    code, (code==200?"OK":"Service Unavailable"), body.size());
  send(fd, head, n, 0); if(!body.empty()) send(fd, body.data(), body.size(), 0);
}

static bool readiness_check(){
  try{
    json jb, body = { {"packed_scores", json::array()} };
    int r=0; long ms=0;
    if (!http_post_json_retry(g_he_base + "/score/batch", body, jb, r, ms)) return false;
    return true;
  }catch(...){ return false; }
}

static bool proxy_eval_keys(const std::string& body, std::string& out){
  try{
    json in = json::parse(body), resp;
    int r=0; long ms=0;
    if (!http_post_json_retry(g_he_base + "/eval-keys", in, resp, r, ms)) {
      json m = {{"event","eval_keys_error"}, {"http_retries", r}};
      std::cout << m.dump() << std::endl;
      return false;
    }
    out = resp.dump();
    json m = {{"event","eval_keys"}, {"http_retries", r}, {"http_ms", ms}};
    std::cout << m.dump() << std::endl;
    return true;
  }catch(...){
    json m = {{"event","eval_keys_parse_error"}};
    std::cout << m.dump() << std::endl;
    return false;
  }
}

static ssize_t find_header_end(const char* buf, ssize_t n){
  for (ssize_t i=0;i+3<n;i++) if (buf[i]=='\r'&&buf[i+1]=='\n'&&buf[i+2]=='\r'&&buf[i+3]=='\n') return i+4;
  return -1;
}

static void healthz_thread(){
  int s = socket(AF_INET, SOCK_STREAM, 0);
  int on=1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  sockaddr_in a{}; a.sin_family=AF_INET; a.sin_addr.s_addr=htonl(INADDR_LOOPBACK); a.sin_port=htons(8080);
  bind(s,(sockaddr*)&a,sizeof(a)); listen(s,8);
  std::cerr << "[Gateway] health server 127.0.0.1:8080\n";

  while (true){
    int c=accept(s,nullptr,nullptr); if(c<0) continue;

    char buf[4096]; int n = recv(c, buf, sizeof(buf)-1, 0);
    if (n<=0){ close(c); continue; } buf[n]='\0';

    char* crlf = strstr(buf, "\r\n");
    size_t linelen = crlf ? (size_t)(crlf - buf) : strlen(buf);
    std::string first(buf, linelen);
    size_t sp1 = first.find(' '), sp2 = (sp1==std::string::npos)?std::string::npos:first.find(' ', sp1+1);
    std::string method = (sp1!=std::string::npos)? first.substr(0, sp1):"";
    std::string path   = (sp1!=std::string::npos && sp2!=std::string::npos)? first.substr(sp1+1, sp2-(sp1+1)):"";

    if (method=="GET" && path=="/healthz") { http_send(c,200,"\"ok\""); close(c); continue; }
    if (method=="GET" && path=="/readyz")  { bool ok = readiness_check(); http_send(c, ok?200:503, ok?"\"ready\"":"\"not ready\""); close(c); continue; }
    if (method=="POST" && path=="/eval-keys") {
      ssize_t head_end = find_header_end(buf, n);
      if (head_end<0){ http_send(c,400,"\"bad header\""); close(c); continue; }
      std::string headers(buf, head_end);
      size_t pos = headers.find("Content-Length:");
      if (pos==std::string::npos){ http_send(c,411,"\"length required\""); close(c); continue; }
      size_t end = headers.find("\r\n", pos);
      std::string v = headers.substr(pos+15, end-pos-15);
      int clen = std::stoi(v);
      std::string body;
      body.assign(buf+head_end, std::min<int>(clen, n - head_end));
      while ((int)body.size() < clen) { char tmp[4096]; int m = recv(c, tmp, sizeof(tmp), 0); if (m<=0) break; body.append(tmp, tmp+m); }
      std::string out;
      if (proxy_eval_keys(body, out)) http_send(c,200, out.empty()?"\"ok\"":out);
      else http_send(c,500,"\"proxy failed\"");
      close(c); continue;
    }
    http_send(c,404,"\"not found\""); close(c);
  }
}

// -------------------- Gateway 实现 --------------------
class GatewayServiceImpl final : public GatewayService::Service {
  Status Search(ServerContext* ctx, const SearchRequest* req, SearchResponse* resp) override {
    auto t0 = std::chrono::steady_clock::now();
    const int L = ENV_L();
    const int P95 = ENV_P95MS();
    const int JIT = ENV_JITTER();
    // === NEW: Detect multimodal mode ===
    std::string mode = "image";  // default

    const auto& md2 = ctx->client_metadata();
    auto it = md2.find("x-query-mode");
    if (it != md2.end()) {
        mode = std::string(it->second.data(), it->second.length());
     }

    // update g_index_base according to mode
    if (mode == "image")       g_index_base = "http://127.0.0.1:18081";
    else if (mode == "text")   g_index_base = "http://127.0.0.1:18086";
    else if (mode == "audio")  g_index_base = "http://127.0.0.1:18085";
    else g_index_base = "http://127.0.0.1:18081"; // fallback
    bool use_parallel = ENV_A_PARALLEL() != 0;
    bool okA = false;

    // === NEW: HMAC 校验（若配置了共享密钥） ===
    std::call_once(g_hmac_once, load_hmac_key_once);
    if (!g_hmac_key.empty()) {
      std::string mac_hex, nonce, ts, keyver_m;
      const auto& md = ctx->client_metadata();
      bool has_mac = meta_get(md, "x-ct-mac", mac_hex);
      bool has_n   = meta_get(md, "x-ct-nonce", nonce);
      bool has_ts  = meta_get(md, "x-ct-ts", ts);
      meta_get(md, "x-ct-keyver", keyver_m); // 可选

      bool mac_ok = false;
      std::string mac_err;

      if (!(has_mac && has_n && has_ts)) {
        mac_err = "mac_missing";
      } else {
        // 检查 ts 窗口（±120s）
        long now = (long)std::time(nullptr);
        long ts_i = 0;
        try { ts_i = std::stol(ts); } catch (...) { mac_err = "ts_bad"; }
        if (mac_err.empty() && std::llabs(now - ts_i) > 120) mac_err = "ts_window";

        // nonce 去重
        if (mac_err.empty() && nonce_seen_or_insert(nonce, 300)) mac_err = "nonce_seen";

        // 组装 mac_msg: ct_q | session_id | clusters_join | key_ver | ts | nonce
        if (mac_err.empty()) {
          std::vector<std::string> clusters(req->cluster_ids().begin(), req->cluster_ids().end());
          std::string join = join_clusters(clusters);
          const std::string& sid = req->session_id();
          const std::string keyver = req->key_ver().empty() ? keyver_m : req->key_ver();

          std::string sep = "|";
          std::string head = sid + sep + join + sep + keyver + sep + ts + sep + nonce;
          // 为避免一次性复制整个消息，分两段 HMAC：先 ct，再 head（等价于拼接）
          HMAC_CTX* hctx = HMAC_CTX_new();
          unsigned char md_buf[EVP_MAX_MD_SIZE]; unsigned int mdlen=0;
          HMAC_Init_ex(hctx, g_hmac_key.data(), (int)g_hmac_key.size(), EVP_sha256(), nullptr);
          HMAC_Update(hctx, reinterpret_cast<const unsigned char*>(req->ct_q().data()), (size_t)req->ct_q().size());
          HMAC_Update(hctx, reinterpret_cast<const unsigned char*>("|"), 1);
          HMAC_Update(hctx, reinterpret_cast<const unsigned char*>(head.data()), head.size());
          HMAC_Final(hctx, md_buf, &mdlen);
          HMAC_CTX_free(hctx);

          // 转 hex
          static const char* H = "0123456789abcdef";
          std::string calc; calc.resize(mdlen*2);
          for (unsigned int i=0;i<mdlen;i++){ calc[i*2]=H[(md_buf[i]>>4)&0xF]; calc[i*2+1]=H[md_buf[i]&0xF]; }

          // 比对（常量时间）
          if (mac_hex.size()==calc.size()) {
            volatile unsigned int diff=0;
            for (size_t i=0;i<calc.size();++i) diff |= (unsigned int)(unsigned char)(mac_hex[i]) ^ (unsigned int)(unsigned char)(calc[i]);
            mac_ok = (diff == 0);
          } else mac_ok = false;

          if (!mac_ok) mac_err = "mac_mismatch";
        }
      }

      if (!mac_err.empty()) {
        // MAC 校验失败：返回定长伪响应，避免 oracle
        const int out_n=ENV_OUTCOUNT(), ct_sz=ENV_CTBYTES(); const uint32_t slots=(uint32_t)ENV_SLOTS();
        for (int i=0;i<out_n;i++){ resp->add_scores_ciphertexts(std::string(ct_sz,'\0')); auto* s=resp->add_pack_shapes(); s->set_batch(1); s->set_slots(slots); }
        // 时间恒定化
        auto t1 = std::chrono::steady_clock::now();
        auto used = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        if (used < P95) std::this_thread::sleep_for(std::chrono::milliseconds(P95 - used));
        if (JIT>0) std::this_thread::sleep_for(std::chrono::milliseconds(rand()%JIT));
        auto t2 = std::chrono::steady_clock::now();
        uint32_t total_ms = (uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
        resp->set_latency_ms(total_ms);

        json m = {{"event","search"}, {"mac_invalid",1}, {"mac_err", mac_err}};
        std::cout << m.dump() << std::endl;
        return Status::OK;
      }
    }

    // 配额
    if (!quota_check(req->session_id())) {
      const int out_n=ENV_OUTCOUNT(), ct_sz=ENV_CTBYTES(); const uint32_t slots=(uint32_t)ENV_SLOTS();
      for (int i=0;i<out_n;i++){ resp->add_scores_ciphertexts(std::string(ct_sz,'\0')); auto* s=resp->add_pack_shapes(); s->set_batch(1); s->set_slots(slots); }
      if (P95>0) std::this_thread::sleep_for(std::chrono::milliseconds(P95));
      if (JIT>0) std::this_thread::sleep_for(std::chrono::milliseconds(rand()%JIT));
      resp->set_latency_ms(P95 + (JIT?rand()%JIT:0));
      json m = {{"event","search"}, {"quota_exceeded",1}};
      std::cout << m.dump() << std::endl;
      return Status::OK;
    }

    // 限流
    bool rateLimited=false;
    if (!rate_allow(req->session_id(), ENV_G_QPS(), ENV_S_QPS(), ENV_BURST())) {
      rateLimited=true;
      const int out_n=ENV_OUTCOUNT(), ct_sz=ENV_CTBYTES(); const uint32_t slots=(uint32_t)ENV_SLOTS();
      for (int i=0;i<out_n;i++){ resp->add_scores_ciphertexts(std::string(ct_sz,'\0')); auto* s=resp->add_pack_shapes(); s->set_batch(1); s->set_slots(slots); }
      if (P95>0) std::this_thread::sleep_for(std::chrono::milliseconds(P95));
      if (JIT>0) std::this_thread::sleep_for(std::chrono::milliseconds(rand()%JIT));
      resp->set_latency_ms(P95 + (JIT?rand()%JIT:0));
      json m = {{"event","search"}, {"rate_limited",1}};
      std::cout << m.dump() << std::endl;
      return Status::OK;
    }

    // 固定 L：不足补假簇；过多截断
    std::vector<std::string> clusters(req->cluster_ids().begin(), req->cluster_ids().end());
    if ((int)clusters.size() < L) {
      uint64_t seed = 0;
      for (auto ch : req->session_id()) seed = seed * 131 + (unsigned char)ch;
      std::mt19937_64 rng(seed ? seed : 0xBADC0DE);
      int need = L - (int)clusters.size();
      std::string policy = ENV_FAKE_POLICY();
      if (policy == "dist_sim" && !g_center_weights.empty()) {
        std::discrete_distribution<int> dist(g_center_weights.begin(), g_center_weights.end());
        for (int i = 0; i < need; i++)
          clusters.push_back("fake_center_" + std::to_string(dist(rng)));
      } else {
        for (int i = 0; i < need; i++)
          clusters.push_back("fake_" + std::to_string(rng() % 1000000));
      }
    } else if ((int)clusters.size() > L) {
      clusters.resize(L);
    }

    // 计划日志
    int origL = req->cluster_ids_size();
    int usedL = (int)clusters.size();
    json m_pre = { {"event","search_plan"}, {"L_orig",origL}, {"L_used",usedL}, {"policy",ENV_FAKE_POLICY()} };
    m_pre["a_mode"] = use_parallel ? "parallel" : "serial";
    std::cout << m_pre.dump() << std::endl;

    // A：候选 + 对角块
    json diag_blocks;
    int rA = 0, rB = 0;
    long msA = 0, msB = 0;
    std::string err_tag;
    uint32_t R = (uint32_t)ENV_R();

    if (use_parallel) {
      okA = call_index_A_parallel(clusters, R, diag_blocks, rA, msA, err_tag);
    } else {
      okA = call_index_A_serial  (clusters, R, diag_blocks, rA, msA, err_tag);
    }

    // 熔断 + B
    bool okB = false;
    std::string err_tag_b;
    if (!circuit_breaker_allow()) {
      okB = false;
      err_tag = "circuit_open";
      json m = {{"event", "circuit_breaker_open"}};
      std::cout << m.dump() << std::endl;
      const int out_n = ENV_OUTCOUNT(), ct_sz = ENV_CTBYTES();
      const uint32_t slots = (uint32_t)ENV_SLOTS();
      for (int i = 0; i < out_n; i++) {
        resp->add_scores_ciphertexts(std::string(ct_sz, '\0'));
        auto* s = resp->add_pack_shapes(); s->set_batch(1); s->set_slots(slots);
      }
    } else {
      okB = call_he_B_and_pack(*req, resp, clusters, diag_blocks, rB, msB, err_tag_b,mode);
      if (okB) circuit_breaker_on_success();
      else {
        circuit_breaker_on_failure();
        if (!err_tag.empty()) err_tag += ",";
        err_tag += err_tag_b;
      }
    }

    // 时间恒定化
    auto t1 = std::chrono::steady_clock::now();
    auto used = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    if (used < P95) std::this_thread::sleep_for(std::chrono::milliseconds(P95 - used));
    if (JIT > 0)   std::this_thread::sleep_for(std::chrono::milliseconds(rand()%JIT));
    auto t2 = std::chrono::steady_clock::now();
    uint32_t total_ms = (uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
    resp->set_latency_ms(total_ms);

    // 指标日志
    json m = {
      {"event","search"},
      {"latency_ms", total_ms},
      {"http_retries_a", rA}, {"http_ms_a", msA},
      {"http_retries_b", rB}, {"http_ms_b", msB},
      {"rate_limited", 0},
      {"quota_exceeded", 0},
      {"a_mode", use_parallel ? "parallel" : "serial"},
      {"A_ok", okA ? 1 : 0}
    };
    if (!err_tag.empty()) m["err"]=err_tag;
    std::cout << m.dump() << std::endl;

    return Status::OK;
  }
};

// -------------------- main --------------------
int main(int argc, char** argv) {
  for (int i=1;i<argc;i++) if (std::string(argv[i])=="-c" && i+1<argc) g_cfg_file = argv[++i];
  if (!g_cfg_file.empty()) load_config(g_cfg_file.c_str());
  validate_config();
  signal(SIGHUP, hup_handler);

  std::string addr = getenv("GATEWAY_LISTEN") ? getenv("GATEWAY_LISTEN") : "0.0.0.0:50052";
  GatewayServiceImpl svc;

  std::thread(healthz_thread).detach();

  ServerBuilder builder;
  builder.SetMaxReceiveMessageSize(256 * 1024 * 1024);
  builder.SetMaxSendMessageSize(256 * 1024 * 1024);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&svc);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "[Gateway] listening on " << addr
            << "  L=" << ENV_L() << " R=" << ENV_R()
            << " ct_bytes=" << ENV_CTBYTES() << " slots=" << ENV_SLOTS()
            << " p95ms=" << ENV_P95MS() << " jitter=" << ENV_JITTER()
            << " retryN=" << ENV_RETRY_N() << " retryBaseMs=" << ENV_RETRY_MS()
            << " retryJitter=" << ENV_RJITTER()
            << " index=" << g_index_base << " he=" << g_he_base
            << std::endl;
  server->Wait();
  return 0;
}

// ======== [PARALLEL] A：候选 + 对角块（并发版） ========
static bool call_index_A_parallel(const std::vector<std::string>& clusters, uint32_t R,
                                  json& out_diag_blocks, int& http_retries_a,
                                  long& http_ms_a, std::string& err_tag) {
  out_diag_blocks = json::object();
  http_retries_a = 0; http_ms_a = 0;
  bool ok = true;

  std::vector<json> per_cluster(clusters.size());
  std::vector<std::thread> ths;
  std::atomic<int> retries_sum{0};
  std::atomic<long> ms_sum{0};

  for (size_t i = 0; i < clusters.size(); ++i) {
    const std::string cid = clusters[i];
    if (!is_digit_only(cid)) {
      per_cluster[i] = json::object({{"__fake__", true}});
      continue;
    }
    ths.emplace_back([&, i, cid]{
      json j; int r = 0; long ms = 0;
      std::string url = g_index_base + "/clusters/" + cid + "/candidates?top=" + std::to_string(R);
      bool ok1 = http_get_json_retry(url, j, r, ms);
      retries_sum += r; ms_sum += ms;
      if (!ok1) j = json::object();
      per_cluster[i] = std::move(j);
    });
  }
  for (auto& t : ths) t.join();
  http_retries_a += retries_sum.load();
  http_ms_a += ms_sum.load();

  json all_candidates = json::array();
  for (auto& j : per_cluster) {
    std::vector<uint64_t> ids;
    if (j.is_object() && j.value("__fake__", false)) {
      ids.resize(R, 0ULL);
    } else if (j.contains("candidates") && j["candidates"].is_array()) {
      for (auto& v : j["candidates"]) {
        if (v.is_object() && v.contains("id"))       ids.push_back(v["id"].get<uint64_t>());
        else if (v.is_number_unsigned())             ids.push_back(v.get<uint64_t>());
        else if (v.is_number_integer()) {
          auto tmp = v.get<long long>(); ids.push_back(tmp < 0 ? 0ULL : (uint64_t)tmp);
        }
      }
    }
    while (ids.size() < R) ids.push_back(0ULL);
    if (ids.size() > R) ids.resize(R);
    for (auto id : ids) all_candidates.push_back(id);
  }

  json req_body = {
    {"candidates", all_candidates},
    {"pack_slots", ENV_SLOTS()},
    {"dim", ENV_DIAG_DIM()}
  };

  json jr; int r = 0; long ms = 0;
  if (!http_post_json_retry(g_index_base + "/diag-blocks", req_body, jr, r, ms)) {
    ok = false; err_tag = "A_diag_blocks";
  } else {
    try {
      std::ofstream ofs("/tmp/diag_blocks.last.json", std::ios::binary);
      ofs << jr.dump(2);
    } catch (...) { /* ignore */ }

    if (jr.contains("blocks")) out_diag_blocks = jr;
    else { ok = false; err_tag = "A_diag_blocks_shape"; }
  }
  http_retries_a += r; http_ms_a += ms;
  return ok;
}
