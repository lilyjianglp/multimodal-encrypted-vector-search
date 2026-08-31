// make_dia1_768.cpp
// 生成 DIA1 多 block 包（每块 128 个对角），并写出 diag_blocks.json 元信息
// 模式：--mode random / --mode csv --csv vectors.csv / --mode from-npy --npy corpus.npy
// 依赖：SEAL-4.1（/usr/local/include/SEAL-4.1, /usr/local/lib/libseal-4.1.a）
// 维度自适配：按 --dim 计算块数 ceil(dim/128)

#include <seal/seal.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <random>
#include <algorithm>
#include <memory>
#include <cstring>
#include <cctype>
#include <cmath>

using namespace seal;
namespace fs = std::filesystem;

// ---------- 简单 JSON 工具 ----------
static void json_escape(std::ostream& os, const std::string& s){
    os << '"';
    for (char c: s){
        switch(c){
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n";  break;
            case '\r': os << "\\r";  break;
            case '\t': os << "\\t";  break;
            default:   os << c;      break;
        }
    }
    os << '"';
}

static void write_u32_le(std::ofstream& os, uint32_t v){
    char b[4];
    b[0]= v      &0xFF;
    b[1]=(v>>8)  &0xFF;
    b[2]=(v>>16) &0xFF;
    b[3]=(v>>24) &0xFF;
    os.write(b,4);
}

// ---------- 候选向量库接口 ----------
struct VecStore {
    virtual bool get(uint64_t id, std::vector<double>& out) = 0;
    virtual ~VecStore() = default;
};

// 稳定“伪随机”向量：按 id 固定种子，均值 0 方差 1，最后单位化
struct RandomDeterministicStore : VecStore {
    size_t dim;
    explicit RandomDeterministicStore(size_t d): dim(d) {}
    bool get(uint64_t id, std::vector<double>& out) override {
        std::mt19937_64 rng(id ? id : 0x9e3779b97f4a7c15ULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        out.resize(dim);
        double ss=0;
        for(size_t i=0;i<dim;i++){ out[i]=nd(rng); ss+=out[i]*out[i]; }
        ss = std::sqrt(ss); if (ss==0) ss=1;
        for(size_t i=0;i<dim;i++) out[i]/=ss;
        return true;
    }
};

// CSV 向量库：文件格式 id,f1,f2,...,fd
struct CSVStore : VecStore {
    size_t dim;
    std::unordered_map<uint64_t, std::vector<double>> mp;
    explicit CSVStore(size_t d): dim(d){}
    bool load(const std::string& path){
        std::ifstream ifs(path);
        if(!ifs) return false;
        std::string line;
        size_t line_no=0, ok=0;
        while (std::getline(ifs, line)){
            line_no++;
            if(line.empty()) continue;
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> cells;
            while(std::getline(ss, cell, ',')) cells.push_back(cell);
            if (cells.size() < dim+1) continue;
            uint64_t id = std::stoull(cells[0]);
            std::vector<double> v(dim);
            for (size_t i=0;i<dim;i++) v[i]=std::stod(cells[1+i]);
            mp.emplace(id, std::move(v)); ok++;
        }
        std::cerr << "[CSVStore] loaded "<< ok <<" rows\n";
        return true;
    }
    bool get(uint64_t id, std::vector<double>& out) override {
        auto it = mp.find(id);
        if (it==mp.end()) return false;
        out = it->second; return true;
    }
};

// NPY（.npy）向量库：float32、C row-major；假设 row_index == id
struct NpyStore : VecStore {
    size_t rows = 0, dim = 0;
    std::vector<float> data; // [rows * dim]

    explicit NpyStore(size_t expect_dim): dim(expect_dim) {}

    static uint16_t rd_u16_le(const unsigned char* p){ return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
    static uint32_t rd_u32_le(const unsigned char* p){ return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }

    static std::string trim(const std::string& s){
        size_t i=0, j=s.size();
        while (i<j && std::isspace((unsigned char)s[i])) i++;
        while (j>i && std::isspace((unsigned char)s[j-1])) j--;
        return s.substr(i, j-i);
    }

    bool load(const std::string& path){
        std::ifstream f(path, std::ios::binary);
        if(!f){ std::cerr << "open fail: "<< path << "\n"; return false; }
        std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), {});
        if (buf.size() < 16){ std::cerr << "npy too small\n"; return false; }
        // magic
        static const unsigned char magic[] = {0x93,'N','U','M','P','Y'};
        if (std::memcmp(buf.data(), magic, sizeof(magic)) != 0){
            std::cerr << "not a .npy file\n"; return false;
        }
        uint8_t ver_major = buf[6], ver_minor = buf[7];
        size_t hlen_off = 8;
        uint32_t header_len = 0;
        size_t header_start = 0;
        if (ver_major == 1){
            header_len = rd_u16_le(buf.data() + hlen_off);
            header_start = 10;
        } else if (ver_major == 2){
            header_len = rd_u32_le(buf.data() + hlen_off);
            header_start = 12;
        } else {
            std::cerr << "unsupported npy version "<< (int)ver_major << "." << (int)ver_minor << "\n";
            return false;
        }
        if (header_start + header_len > buf.size()){
            std::cerr << "npy header truncated\n"; return false;
        }
        std::string header((const char*)buf.data() + header_start, header_len);
        // parse dict: {'descr': '<f4', 'fortran_order': False, 'shape': (N, D), }
        auto find_kv = [&](const std::string& key)->std::string{
            // very naive parse: find "'key':" then read until next comma/bracket
            std::string pat = "'" + key + "':";
            size_t k = header.find(pat);
            if (k == std::string::npos) return "";
            size_t vstart = k + pat.size();
            // skip spaces
            while (vstart < header.size() && std::isspace((unsigned char)header[vstart])) vstart++;
            // collect until comma or }
            size_t vend = vstart;
            int paren=0, bracket=0;
            bool in_str=false;
            while (vend < header.size()){
                char c = header[vend];
                if (c=='\'' || c=='"') { // toggle string (very rough)
                    in_str = !in_str;
                } else if (!in_str) {
                    if (c=='(') paren++;
                    else if (c==')') paren--;
                    else if (c=='[') bracket++;
                    else if (c==']') bracket--;
                    else if (c==',' && paren==0 && bracket==0) break;
                    else if (c=='}' && paren==0 && bracket==0) break;
                }
                vend++;
            }
            return trim(header.substr(vstart, vend - vstart));
        };

        std::string descr = find_kv("descr");
        std::string fortran = find_kv("fortran_order");
        std::string shape = find_kv("shape");
        // descr is quoted, e.g. '<f4'
        auto unquote = [&](std::string s){
            s = trim(s);
            if (!s.empty() && (s.front()=='\'' || s.front()=='"') && s.back()==s.front()){
                return s.substr(1, s.size()-2);
            }
            return s;
        };
        descr = unquote(descr);
        bool fortran_order = (fortran.find("True") != std::string::npos);

        if (descr != "<f4" && descr != "|f4" && descr != "f4"){
            std::cerr << "npy descr not float32: " << descr << "\n"; return false;
        }
        if (fortran_order){
            std::cerr << "npy Fortran order not supported\n"; return false;
        }
        // parse shape like (N, D) or (N, D,)
        size_t n1 = shape.find('('), n2 = shape.find(')');
        if (n1==std::string::npos || n2==std::string::npos || n2<=n1+1){
            std::cerr << "bad npy shape: " << shape << "\n"; return false;
        }
        std::string inner = trim(shape.substr(n1+1, n2-n1-1));
        // split by comma
        std::vector<size_t> dims;
        {
            std::stringstream ss(inner);
            std::string tok;
            while (std::getline(ss, tok, ',')){
                tok = trim(tok);
                if (tok.empty()) continue;
                dims.push_back((size_t)std::stoul(tok));
            }
        }
        if (dims.size() < 2){
            std::cerr << "npy shape need at least 2 dims, got: " << inner << "\n"; return false;
        }
        rows = dims[0];
        size_t d = dims[1];
        if (dim != 0 && d != dim){
            std::cerr << "[WARN] npy dim="<< d <<" but --dim="<< dim <<", will use npy dim\n";
            dim = d; // 以文件为准
        }

        size_t data_off = header_start + header_len;
        size_t need_bytes = rows * dim * sizeof(float);
        if (data_off + need_bytes > buf.size()){
            std::cerr << "npy data truncated: need "<< need_bytes <<", have "<< (buf.size()-data_off) << "\n";
            return false;
        }

        data.resize(rows * dim);
        std::memcpy(data.data(), buf.data() + data_off, need_bytes);
        std::cerr << "[NpyStore] loaded rows="<< rows <<" dim="<< dim << "\n";
        return true;
    }

    bool get(uint64_t id, std::vector<double>& out) override {
        if (id >= rows){
            // 不存在的 id 返回 false
            return false;
        }
        out.resize(dim);
        const float* row = &data[ (size_t)id * dim ];
        for (size_t i=0;i<dim;i++) out[i] = (double)row[i];
        return true;
    }
};

// 读取 ids：支持 txt(每行一个) 或 JSON 数组
static bool load_ids(const std::string& path, std::vector<uint64_t>& ids){
    std::ifstream f(path);
    if(!f) return false;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto trim = [](std::string x){
        x.erase(x.begin(), std::find_if(x.begin(), x.end(), [](unsigned char c){return !std::isspace(c);} ));
        x.erase(std::find_if(x.rbegin(), x.rend(), [](unsigned char c){return !std::isspace(c);} ).base(), x.end());
        return x;
    };
    s = trim(s);
    if (!s.empty() && s.front()=='['){
        uint64_t cur=0; bool in_num=false;
        for (char c: s){
            if (c>='0' && c<='9'){ cur = in_num? (cur*10 + (c-'0')) : (c-'0'); in_num=true; }
            else {
                if (in_num){ ids.push_back(cur); in_num=false; cur=0; }
            }
        }
        if (in_num) ids.push_back(cur);
        return true;
    }else{
        std::stringstream ss(s);
        std::string line;
        while(std::getline(ss, line)){
            // 去除注释和空白
            auto p = line.find('#');
            if (p != std::string::npos) line = line.substr(0, p);
            line = trim(line);
            if (line.empty()) continue;
            ids.push_back(std::stoull(line));
        }
        return true;
    }
}

// 核心：构建一个 block（128 对角），写 DIA1
static int build_block_pack(const SEALContext& ctx,
                            const std::vector<uint64_t>& slot_ids, // 4096
                            VecStore& store,
                            size_t dim,          // e.g. 512 / 768
                            size_t block_offset, // 0,128,256,...
                            const std::string& out_path,
                            bool full_bsgs,
                            size_t baby_size)
{
    CKKSEncoder encoder(ctx);
    const double scale = std::pow(2.0, 40);

    // 预取候选
    std::vector<std::vector<double>> cand(slot_ids.size());
    for (size_t s=0; s<slot_ids.size(); ++s) {
        if (!store.get(slot_ids[s], cand[s])) {
            // 不存在则置零
            cand[s].assign(dim, 0.0);
        } else if (cand[s].size() != dim) {
            // 尺寸不一致也兜底
            std::vector<double> tmp(dim, 0.0);
            for (size_t i=0;i<std::min(dim, cand[s].size()); ++i) tmp[i] = cand[s][i];
            cand[s].swap(tmp);
        }
    }

    std::ofstream os(out_path, std::ios::binary);
    if(!os){ std::cerr << "open fail: "<< out_path <<"\n"; return 1; }

    os.write("DIA1", 4);
    // 本块实际对角数：可能不足 128（末块）
    const uint32_t COUNT = (uint32_t)std::min<size_t>(128, (dim > block_offset ? (dim - block_offset) : 0));
    write_u32_le(os, COUNT);

    for (uint32_t t=0; t<COUNT; ++t) {
        size_t diag = block_offset + t;
        std::vector<double> diag_vec(slot_ids.size());
        for (size_t s=0; s<slot_ids.size(); ++s) {
            // q is repeated every `dim` slots. After rotating q by `diag`,
            // slot s contains q[(s + diag) % dim], so the matching matrix
            // diagonal must contain the same coordinate of candidate s.
            size_t j = (s + diag) % dim;
            diag_vec[s] = cand[s][j];
        }
        if (full_bsgs) {
            const size_t giant = (diag / baby_size) * baby_size;
            if (giant != 0) {
                std::vector<double> adjusted(diag_vec.size());
                const size_t shift = giant % diag_vec.size();
                // Store RotPlain(P_diag, -giant). The online evaluator first
                // accumulates a baby group, then applies one Rot(+giant).
                for (size_t s = 0; s < diag_vec.size(); ++s) {
                    adjusted[s] = diag_vec[(s + diag_vec.size() - shift) % diag_vec.size()];
                }
                diag_vec.swap(adjusted);
            }
        }
        Plaintext pt;
        encoder.encode(diag_vec, scale, pt);
        std::stringstream ss; pt.save(ss);
        std::string bytes = ss.str();
        write_u32_le(os, (uint32_t)bytes.size());
        os.write(bytes.data(), (std::streamsize)bytes.size());
    }
    return 0;
}

static void usage(const char* prog){
    std::cerr <<
    "Usage:\n"
    "  " << prog << " --context context.seal --ids ids.txt  --dim 768 --outdir data/diag/D0768 --mode random\n"
    "  " << prog << " --context context.seal --ids ids.json --dim 768 --outdir data/diag/D0768 --mode csv   --csv vectors.csv\n"
    "  " << prog << " --context context.seal --ids ids.txt  --dim 512 --outdir data/diag/D0512 --mode from-npy --npy corpus.npy [--full-bsgs] [--bsgs-baby 32]\n";
}

int main(int argc, char** argv){
    std::string context_path, ids_path, outdir="out_diag";
    std::string mode="random", csv_path, npy_path;
    size_t dim = 768;
    size_t baby_size = 32;
    bool full_bsgs = false;

    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        auto need = [&](const char* name)->std::string{
            if (i+1>=argc){ usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        if (a=="--context") context_path = need("--context");
        else if (a=="--ids") ids_path    = need("--ids");
        else if (a=="--outdir") outdir   = need("--outdir");
        else if (a=="--dim") dim         = (size_t)std::stoul(need("--dim"));
        else if (a=="--mode") mode       = need("--mode");
        else if (a=="--csv") csv_path    = need("--csv");
        else if (a=="--npy") npy_path    = need("--npy");
        else if (a=="--full-bsgs") full_bsgs = true;
        else if (a=="--bsgs-baby") baby_size = (size_t)std::stoul(need("--bsgs-baby"));
        else { usage(argv[0]); return 1; }
    }
    if (context_path.empty() || ids_path.empty()){ usage(argv[0]); return 1; }
    fs::create_directories(outdir);

    // 1) 还原 SEALContext
    EncryptionParameters parms;
    {
        std::ifstream f(context_path, std::ios::binary);
        if(!f){ std::cerr << "open fail: "<< context_path <<"\n"; return 1; }
        std::stringstream ss; ss << f.rdbuf();
        parms.load(ss);
    }
    SEALContext ctx(parms, /*expand_mod_chain=*/true);

    // 2) 读取 4096 个 slot_ids
    std::vector<uint64_t> slot_ids;
    if (!load_ids(ids_path, slot_ids)){ std::cerr << "load ids fail: "<< ids_path <<"\n"; return 1; }
    if (slot_ids.size() != 4096){
        std::cerr << "[WARN] ids count="<< slot_ids.size() <<" (expected 4096), will pad/truncate\n";
        if (slot_ids.size() < 4096) slot_ids.resize(4096, 0);
        else slot_ids.resize(4096);
    }
    if (baby_size == 0) {
        std::cerr << "--bsgs-baby must be positive\n";
        return 1;
    }

    // 3) 准备向量库
    std::unique_ptr<VecStore> store;
    if (mode=="random"){
        store = std::make_unique<RandomDeterministicStore>(dim);
    }else if (mode=="csv"){
        auto csvs = std::make_unique<CSVStore>(dim);
        if (csv_path.empty()){ std::cerr << "need --csv\n"; return 1; }
        if (!csvs->load(csv_path)){ std::cerr << "load csv fail: "<< csv_path <<"\n"; return 1; }
        store = std::move(csvs);
    }else if (mode=="from-npy"){
        if (npy_path.empty()){ std::cerr << "need --npy\n"; return 1; }
        auto npys = std::make_unique<NpyStore>(dim);
        if (!npys->load(npy_path)){ std::cerr << "load npy fail: "<< npy_path <<"\n"; return 1; }
        // 用文件内实际维度覆盖 dim（NpyStore.load 中已同步）
        dim = npys->dim;
        store = std::move(npys);
    }else{
        std::cerr << "unknown --mode: "<< mode <<"\n"; return 1;
    }
    if (full_bsgs && (slot_ids.size() % dim) != 0) {
        std::cerr << "full BSGS dense packing requires slot count divisible by dim: slots="
                  << slot_ids.size() << " dim=" << dim << "\n";
        return 1;
    }

    // 4) 块数自适应：ceil(dim/128)
    const size_t blk_cnt = (dim + 127) / 128;

    std::vector<std::string> block_paths;
    for (size_t i=0;i<blk_cnt;i++){
        const size_t off = i * 128;
        char name[64]; std::snprintf(name, sizeof(name), "blk-%06zu.dia", off);
        std::string path = (fs::path(outdir)/name).string();
        std::cerr << "[build] " << path << "\n";
        if (build_block_pack(ctx, slot_ids, *store, dim, off, path,
                             full_bsgs, baby_size)!=0) return 1;
        block_paths.push_back(path);
    }

    // 5) 写 diag_blocks.json（给 /diag-blocks 用）
    std::string json_path = (fs::path(outdir)/"diag_blocks.json").string();
    std::ofstream jf(json_path);
    jf << "{\n  \"blocks\": [\n";
    for (size_t i=0;i<blk_cnt;i++){
        const size_t off = i*128;
        char blkid[64]; std::snprintf(blkid, sizeof(blkid), "blk-%06zu", off);
        jf << "    {\n";
        jf << "      \"block_id\": "; json_escape(jf, blkid); jf << ",\n";
        jf << "      \"mmap_path\": "; json_escape(jf, block_paths[i]); jf << ",\n";
        jf << "      \"layout\": {\n";
        jf << "        \"slots\": 4096,\n";
        jf << "        \"stride\": 4096,\n";
        jf << "        \"diag_offsets\": [";
        // 该块的实际对角数（末块可能 <128）
        size_t cnt = std::min<size_t>(128, (dim>off ? (dim-off) : 0));
        for (size_t t=0;t<cnt;t++){ if (t) jf << ","; jf << (off+t); }
        jf << "],\n";
        jf << "        \"packing\": \""
           << (full_bsgs ? "offset-major-bsgs-v1" : "offset-major") << "\",\n";
        jf << "        \"bsgs_baby\": " << baby_size << ",\n";
        jf << "        \"plaintext_pre_rotated\": "
           << (full_bsgs ? "true" : "false") << ",\n";
        jf << "        \"poly_modulus_degree\": 8192,\n";
        jf << "        \"scale\": 1099511627776.0,\n";
        jf << "        \"level\": 0\n";
        jf << "      },\n";
        jf << "      \"slot_ids\": [";
        for (size_t s=0;s<slot_ids.size();++s){ if (s) jf << ","; jf << slot_ids[s]; }
        jf << "]\n";
        jf << "    }";
        jf << (i+1<blk_cnt ? ",\n" : "\n");
    }
    jf << "  ]\n}\n";
    std::cerr << "[OK] diag_blocks.json -> " << json_path << "\n";
    return 0;
}
