// server_main.cpp  — HeCompute gRPC 入口（RAW dump/返回恒定化修复版）

#include <grpcpp/grpcpp.h>
#include "hecompute.pb.h"
#include "hecompute.grpc.pb.h"

#include <seal/seal.h>
#include <seal/serialization.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cstdio>

#include "hecompute_service.hpp"

namespace pb = hecompute::v1;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using namespace seal;

// ----------------- 运行时配置 -----------------
struct Config {
    std::string listen = "0.0.0.0:18082";
    size_t N = 8192;
    std::vector<int> mod_bits = {60, 40, 40, 60};
    uint32_t out_ct_count = 8;    // 仅作为“期望个数”的默认值；实际以 core 返回为准
    uint32_t pack_slots   = 4096;
    int threads = 8;
};

static std::string getenv_s(const char* k, const std::string& d){ if (const char* v=getenv(k)) return v; return d; }
static uint32_t getenv_u32(const char* k, uint32_t d){ if (const char* v=getenv(k)) return (uint32_t)strtoul(v,nullptr,10); return d; }
static int getenv_i(const char* k, int d){ if (const char* v=getenv(k)) return atoi(v); return d; }

static Config LoadCfg(){
    Config c;
    c.listen       = getenv_s("HE_LISTEN", c.listen);
    // 【注意】不再读取/使用 HE_OUT_CT_BYTES
    c.out_ct_count = getenv_u32("HE_OUT_CT_COUNT", c.out_ct_count);
    c.pack_slots   = getenv_u32("HE_PACK_SLOTS",   c.pack_slots);
    c.threads      = getenv_i ("HE_THREADS",       c.threads);
    return c;
}

// ----------------- Key 缓存 -----------------
struct ClientKey { GaloisKeys gal; RelinKeys relin; };
class KeyCache {
    std::mutex m_;
    std::unordered_map<std::string, ClientKey> mp_;
    static std::string K(const std::string& a,const std::string& b){ return a+"\x1f"+b; }
public:
    void Put(const std::string& id,const std::string& ver, ClientKey&& k){
        std::lock_guard<std::mutex> lk(m_); mp_[K(id,ver)] = std::move(k);
    }
    bool Get(const std::string& id,const std::string& ver, ClientKey& out){
        std::lock_guard<std::mutex> lk(m_);
        auto it = mp_.find(K(id,ver));
        if (it==mp_.end()) return false;
        out = it->second; return true;
    }
};

// ----------------- gRPC 实现 -----------------
class HeComputeRpc final : public pb::HeComputeService::Service {
    Config      cfg_;
    SEALContext ctx_;
    ::HeComputeService core_;
    KeyCache   cache_;

    template<class T>
    bool loadSeal(T& obj, const SEALContext& ctx, const std::string& bytes){
        std::stringstream ss(std::string(bytes.data(), bytes.size()));
        try{ obj.load(ctx, ss); return true; }catch(...){ return false; }
    }
    template<class T>
    std::string saveSeal(const T& obj, seal::compr_mode_type mode = seal::compr_mode_type::none){
        std::ostringstream os;
        obj.save(os, mode);
        return os.str();
    }

    ::DiagBlock toCore(const pb::DiagBlock& in){
        ::DiagBlock out;
        out.block_id    = in.block_id();
        out.snapshot_id = in.snapshot_id();
        out.layout.slots = in.layout().slots();
        out.layout.stride= in.layout().stride();
        out.layout.packing = in.layout().packing();
        out.layout.poly_modulus_degree = in.layout().poly_modulus_degree();
        out.layout.scale = in.layout().scale();
        out.layout.level = in.layout().level();
        out.layout.diag_offsets.clear();
        for (auto v: in.layout().diag_offsets()) out.layout.diag_offsets.push_back((int)v);
        out.diag_plaintexts.clear();
        out.diag_plaintexts.reserve(in.diag_plaintexts_size());
        for (const auto& b: in.diag_plaintexts()){
            Plaintext pt;
            if (!loadSeal(pt, ctx_, b)) out.diag_plaintexts.emplace_back();
            else out.diag_plaintexts.emplace_back(std::move(pt));
        }
        return out;
    }

    // 辅助：打印 Ciphertext 关键信息（调试）
    static void log_ct_info(std::ostream& os, const Ciphertext& ct, const std::string& tag){
        os << "[HeComputeRpc] " << tag
           << " size=" << ct.size()
           << " polyN=" << ct.poly_modulus_degree()
           << " is_ntt=" << (ct.is_ntt_form() ? 1 : 0)
           << " parms_id=(" << ct.parms_id()[0] << "," << ct.parms_id()[1]
           << "," << ct.parms_id()[2] << "," << ct.parms_id()[3] << ")"
           << std::endl;
    }

public:
    HeComputeRpc(const Config& cfg, const SEALContext& ctx)
        : cfg_(cfg), ctx_(ctx), core_(ctx) {
        core_.SetNumThreads(cfg_.threads);
    }

    Status EvalKeys(ServerContext*, const pb::EvalKeysRequest* req, pb::EvalKeysReply* rsp) override {
        ClientKey k;
        if (!loadSeal(k.gal,   ctx_, req->galois()) ||
            !loadSeal(k.relin, ctx_, req->relin())) {
            rsp->set_ok(false); rsp->set_msg("bad galois/relin");
            return Status::OK;
        }
        core_.EvalKeys(req->client_id(), req->key_ver(), k.gal, k.relin);
        cache_.Put(req->client_id(), req->key_ver(), std::move(k));
        rsp->set_ok(true); rsp->set_msg("ok");
        return Status::OK;
    }

    Status ScoreBatch(ServerContext*, const pb::ScoreBatchRequest* req, pb::ScoreBatchReply* rsp) override {
        ClientKey k;
        if (!cache_.Get(req->client_id(), req->key_ver(), k))
            return Status(grpc::StatusCode::FAILED_PRECONDITION, "keys not found");

        Ciphertext ct_q;
        if (!loadSeal(ct_q, ctx_, req->ct_q()))
            return Status(grpc::StatusCode::INVALID_ARGUMENT, "bad ct_q");

        // 打印查询密文信息，便于对齐 level/scale/parms_id
        log_ct_info(std::cerr, ct_q, "ct_q");

        ::BSGSPlan plan;
        for (auto v: req->bsgs().baby())  plan.baby.push_back((int)v);
        for (auto v: req->bsgs().giant()) plan.giant.push_back((int)v);

        // 基本合法性：BSGS 不应为空
        if (plan.baby.empty() || plan.giant.empty()) {
            std::cerr << "[HeComputeRpc][ERR] invalid BSGS plan: baby=" << plan.baby.size()
                      << " giant=" << plan.giant.size() << std::endl;
            return Status(grpc::StatusCode::FAILED_PRECONDITION, "invalid BSGS plan (empty)");
        }

        std::vector<::DiagBlock> blocks;
        blocks.reserve(req->blocks_size());

        size_t total_pts = 0, non_empty_pts = 0;
        for (int i=0; i<req->blocks_size(); ++i) {
            auto &b = req->blocks(i);
            size_t pts = static_cast<size_t>(b.diag_plaintexts_size());
            total_pts += pts;

            auto core_b = toCore(b);
            for (const auto &pt : core_b.diag_plaintexts) {
                if (!pt.is_zero() && pt.coeff_count() > 0) ++non_empty_pts;
            }

            std::cerr << "[HeComputeRpc] blk#" << i
                      << " id=" << b.block_id()
                      << " slots=" << b.layout().slots()
                      << " stride="<< b.layout().stride()
                      << " polyN=" << b.layout().poly_modulus_degree()
                      << " diag_cnt=" << pts
                      << " non_empty=" << non_empty_pts << std::endl;

            if (b.layout().diag_offsets_size() > 0 &&
                static_cast<size_t>(b.layout().diag_offsets_size()) != pts) {
                std::cerr << "[HeComputeRpc][WARN] diag_offsets("
                          << b.layout().diag_offsets_size()
                          << ") != diag_plaintexts(" << pts << ")\n";
            }
            blocks.emplace_back(std::move(core_b));
        }
        std::cerr << "[HeComputeRpc] total blocks=" << req->blocks_size()
                  << " total_diag_pts=" << total_pts << std::endl;

        // 明文全空，直接 fail，避免“悄悄返回空密文”
        if (total_pts == 0 || non_empty_pts == 0) {
            std::cerr << "[HeComputeRpc][ERR] all diag plaintexts are empty (total_pts="
                      << total_pts << ", non_empty=" << non_empty_pts << ")\n";
            return Status(grpc::StatusCode::FAILED_PRECONDITION, "diag plaintexts empty/invalid");
        }

        ::ScoreBatchRequest creq;
        creq.client_id = req->client_id();
        creq.key_ver   = req->key_ver();
        creq.ct_q      = std::move(ct_q);
        creq.blocks    = std::move(blocks);
        creq.bsgs      = std::move(plan);
        creq.scale     = req->scale();

        auto t0 = std::chrono::high_resolution_clock::now();
        ::ScoreBatchReply crep = core_.ScoreBatch(creq);
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // 若 core 产出为空或含空密文，直接报错并给出统计
        size_t empty_ct = 0;
        for (size_t i = 0; i < crep.scores_ciphertexts.size(); ++i) {
            if (crep.scores_ciphertexts[i].size() == 0) ++empty_ct;
        }
        if (crep.scores_ciphertexts.empty() || empty_ct == crep.scores_ciphertexts.size()) {
            std::cerr << "[HeComputeRpc][ERR] core returned empty ciphertexts: "
                      << "count=" << crep.scores_ciphertexts.size()
                      << " empty=" << empty_ct << std::endl;
            return Status(grpc::StatusCode::FAILED_PRECONDITION, "core returned empty ciphertexts");
        }

        // ===== RAW dump（不压缩，完整保存） =====
        if (const char* dump_dir = std::getenv("HE_DUMP_RAW_DIR")) {
            try {
                std::error_code ec;
                std::filesystem::create_directories(dump_dir, ec);
                for (size_t i = 0; i < crep.scores_ciphertexts.size(); ++i) {
                    const auto &ct = crep.scores_ciphertexts[i];
                    std::string bytes = saveSeal(ct, seal::compr_mode_type::none);
                    char name[64];
                    std::snprintf(name, sizeof(name), "scores_%02zu.bin", i);
                    const std::string path = std::string(dump_dir) + "/" + name;
                    std::ofstream ofs(path, std::ios::binary);
                    if (!ofs) throw std::runtime_error("open failed: " + path);
                    ofs.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                    ofs.close();
                    std::cerr << "[HeComputeRpc] RAW saved " << path
                              << " bytes=" << bytes.size() << std::endl;
                }
                fprintf(stderr, "[HeComputeRpc] RAW dump -> %s (count=%zu)\n",
                        dump_dir, crep.scores_ciphertexts.size());
                fflush(stderr);
            } catch (const std::exception& e) {
                std::cerr << "[HeComputeRpc] dump raw failed: " << e.what() << std::endl;
            }
        }

        // ===== gRPC 返回：逐个真实回传（不再 pad/trim） =====
        const uint32_t SLOTS = req->pack_slots() ? req->pack_slots() : cfg_.pack_slots;
        std::cerr << "[HeComputeRpc] reply cts=" << crep.scores_ciphertexts.size() << std::endl;

        for (size_t i = 0; i < crep.scores_ciphertexts.size(); ++i) {
            const auto &ct = crep.scores_ciphertexts[i];
            std::string bytes = saveSeal(ct, seal::compr_mode_type::none);
            log_ct_info(std::cerr, ct, "reply.ct#" + std::to_string(i));
            std::cerr << "[HeComputeRpc] reply.ct#" << i << " serialized_bytes=" << bytes.size() << std::endl;

            rsp->add_scores_ciphertexts(std::move(bytes));
            auto* ps = rsp->add_pack_shapes();
            ps->set_batch(1);
            ps->set_slots(SLOTS);
        }

        rsp->mutable_telemetry()->set_lat_us(us);
        rsp->mutable_telemetry()->set_rot_cnt(crep.telemetry.rot_cnt);
        rsp->mutable_telemetry()->set_mul_cnt(crep.telemetry.mul_cnt);
        return Status::OK;
    }
};

int main(){
    Config cfg = LoadCfg();
    std::cout << "[HeCompute] build=" << __DATE__ << " " << __TIME__ << std::endl;

    // 优先从文件加载 context.seal（与客户端一致）
    std::string ctx_path = getenv_s("HE_CONTEXT", "");
    if (ctx_path.empty()) ctx_path = "context.seal";

    EncryptionParameters parms(seal::scheme_type::ckks);
    SEALContext context(parms, /*expand_mod_chain=*/true); // 临时占位
    bool loaded_from_file = false;

    {
        std::ifstream ifs(ctx_path, std::ios::binary);
        if (ifs.is_open()) {
            std::stringstream ss; ss << ifs.rdbuf();
            try {
                parms.load(ss);
                context = SEALContext(parms, /*expand_mod_chain=*/true);
                loaded_from_file = true;
                std::cout << "[HeCompute] Loaded context from file: " << ctx_path << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[HeCompute] Failed to load context.seal (" << ctx_path
                          << "): " << e.what() << " -> fallback to compiled defaults" << std::endl;
            }
        } else {
            std::cerr << "[HeCompute] context.seal not found at " << ctx_path
                      << " -> fallback to compiled defaults" << std::endl;
        }
    }

    if (!loaded_from_file) {
        parms = EncryptionParameters(seal::scheme_type::ckks);
        parms.set_poly_modulus_degree(cfg.N);
        parms.set_coeff_modulus(CoeffModulus::Create(cfg.N, cfg.mod_bits));
        context = SEALContext(parms, /*expand_mod_chain=*/true);
    }

    auto &cm = parms.coeff_modulus();
    std::cout << "[HeCompute] poly_modulus_degree=" << parms.poly_modulus_degree()
              << "  coeff_modulus_size=" << cm.size() << "  bits=[";
    for (size_t i=0;i<cm.size();++i){ if (i) std::cout<<","; std::cout<< cm[i].bit_count(); }
    std::cout << "]" << (loaded_from_file ? "  (from file)\n" : "  (fallback)\n");

    HeComputeRpc service(cfg, context);

    const char* raw = std::getenv("HE_DUMP_RAW_DIR");
    std::cout << "[HeCompute] HE_DUMP_RAW_DIR=" << (raw ? raw : "<unset>") << std::endl;

    // 若用户仍设置了 HE_OUT_CT_BYTES，给出警告（但不使用）
    if (std::getenv("HE_OUT_CT_BYTES")) {
        std::cerr << "[HeCompute][WARN] HE_OUT_CT_BYTES is set but ignored. "
                  << "Server does NOT pad/trim ciphertext bytes anymore.\n";
    }

    ServerBuilder b;
    b.SetMaxReceiveMessageSize(256 * 1024 * 1024);
    b.SetMaxSendMessageSize(256 * 1024 * 1024);
    b.AddListeningPort(cfg.listen, grpc::InsecureServerCredentials());
    b.RegisterService(&service);

    std::unique_ptr<Server> server(b.BuildAndStart());
    std::cout << "[HeCompute] listening on " << cfg.listen
              << " out_ct_count=" << cfg.out_ct_count
              << " pack_slots=" << cfg.pack_slots
              << " threads=" << cfg.threads << std::endl;

    server->Wait();
    return 0;
}

