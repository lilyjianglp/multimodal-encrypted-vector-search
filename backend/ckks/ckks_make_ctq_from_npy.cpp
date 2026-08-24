#include <seal/seal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <cstdint>

using namespace seal;
namespace fs = std::filesystem;

// 读取 .npy (float32, C-order, 1D 或 2D) 的极简函数（只支持 little-endian）
static std::vector<float> load_npy_f32(const std::string &path, size_t &rows, size_t &cols){
    std::ifstream f(path, std::ios::binary);
    if(!f) throw std::runtime_error("open fail: " + path);
    // 读取 header
    char magic[6]; f.read(magic,6);
    if(std::string(magic,6)!="\x93NUMPY") throw std::runtime_error("bad magic");
    uint8_t v_major, v_minor; f.read((char*)&v_major,1); f.read((char*)&v_minor,1);
    uint16_t header_len; f.read((char*)&header_len,2);
    std::string header(header_len, '\0'); f.read(header.data(), header_len);
    // 解析形状
    auto pos = header.find("shape");
    auto l = header.find('(', pos); auto r = header.find(')', l);
    auto inside = header.substr(l+1, r-l-1);
    // 形如 "512" 或 "N, 512"
    rows = 1; cols = 0;
    {
        std::stringstream ss(inside);
        std::string a;
        if(std::getline(ss,a,',')){
            size_t t = 0;
            while(t<a.size() && isspace((unsigned char)a[t])) ++t;
            rows = (size_t)std::stoul(a.substr(t));
            if(std::getline(ss,a,',')){
                size_t t2=0; while(t2<a.size() && isspace((unsigned char)a[t2])) ++t2;
                cols = (size_t)std::stoul(a.substr(t2));
            }else{
                cols = rows; rows = 1; // 1D
            }
        }
    }
    // 读取数据
    std::vector<float> buf(rows*cols);
    f.read((char*)buf.data(), (std::streamsize)(buf.size()*sizeof(float)));
    if(!f) throw std::runtime_error("short read on data");
    return buf;
}

int main(int argc, char** argv){
    if (argc < 5){
        std::cerr << "Usage:\n  " << argv[0]
                  << " --context context.seal --pk pk.bin --npy query.npy [--dim 512] [--out ct_q.bin]\n";
        return 1;
    }
    std::string context_path, pk_path, npy_path, out_path="ct_q.bin";
    size_t dim = 512;
    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        auto need = [&](const char* name){ if(i+1>=argc){ throw std::runtime_error(std::string("missing ")+name); } return std::string(argv[++i]); };
        if(a=="--context") context_path = need("--context");
        else if(a=="--pk") pk_path = need("--pk");
        else if(a=="--npy") npy_path = need("--npy");
        else if(a=="--out") out_path = need("--out");
        else if(a=="--dim") dim = (size_t)std::stoul(need("--dim"));
        else { std::cerr << "Unknown arg: " << a << "\n"; return 1; }
    }

    // 1) 还原 SEALContext
    EncryptionParameters parms;
    {
        std::ifstream f(context_path, std::ios::binary);
        if(!f){ std::cerr << "open fail: " << context_path << "\n"; return 1; }
        std::stringstream ss; ss << f.rdbuf();
        parms.load(ss);
    }
    SEALContext ctx(parms, true);

    // 2) 读取公钥
    PublicKey pk;
    {
        std::ifstream f(pk_path, std::ios::binary);
        if(!f){ std::cerr << "open fail: " << pk_path << "\n"; return 1; }
        std::stringstream ss; ss << f.rdbuf();
        pk.load(ctx, ss);
    }
    Encryptor enc(ctx, pk);
    CKKSEncoder encoder(ctx);

    // 3) 读取 npy 向量
    size_t rows=0, cols=0;
    auto buf = load_npy_f32(npy_path, rows, cols); // rows==1, cols==dim 或 rows==N, cols==dim（取第0行）
    if(rows==1){ /* ok */ }
    else if(rows>1){ /* 取第0行 */ buf.resize(cols); }
    if(cols != dim && buf.size() != dim){
        std::cerr << "npy dim mismatch: got " << (cols?cols:buf.size()) << ", expect " << dim << "\n";
        return 1;
    }

    // 4) 平铺到 4096 槽（slots=4096, N=8192）
    size_t slots = 4096;
    std::vector<double> plain(slots);
    for(size_t i=0;i<slots;i++) plain[i] = (double)buf[i % dim];

    // 5) encode + encrypt
    const double SCALE40 = std::pow(2.0, 40);
    Plaintext pt;
    encoder.encode(plain, SCALE40, pt);
    Ciphertext ct;
    enc.encrypt(pt, ct);

    // 6) 保存 ct_q.bin
    std::ofstream of(out_path, std::ios::binary);
    ct.save(of);
    std::cout << "OK: wrote " << out_path << " (scale=" << ct.scale() << ")\n";
    return 0;
}
