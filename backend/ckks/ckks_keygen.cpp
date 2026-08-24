// ckks_keygen.cpp — 只做 KeyGen，不生成 demo 查询密文
// 默认参数：polyN=8192, coeff_mod_bits=[60,40,40,60], scale=2^40
// 可选：--outdir <dir>
//
// 编译（SEAL 4.1 安装在 /usr/local）:
//   g++ -O2 -std=c++17 ckks_keygen.cpp -o ckks_keygen \
//       -I/usr/local/include/SEAL-4.1 -L/usr/local/lib -lseal-4.1
//
// 运行：
//   ./ckks_keygen                # 输出到当前目录
//   ./ckks_keygen --outdir keys  # 输出到 keys/ 目录

#include <seal/seal.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <cmath>

using namespace std;
using namespace seal;
namespace fs = std::filesystem;

static void save_bytes(const fs::path &p, stringstream &ss) {
    fs::create_directories(p.parent_path());
    ofstream os(p, ios::binary);
    os << ss.rdbuf();
}

int main(int argc, char** argv) {
    try {
        // ---- 解析参数 ----
        fs::path outdir = fs::current_path();
        for (int i = 1; i < argc; ++i) {
            string a = argv[i];
            auto need = [&](const char* name)->const char*{
                if (i+1 >= argc) { cerr << "missing arg after " << name << "\n"; std::exit(1); }
                return argv[++i];
            };
            if (a == "--outdir") outdir = fs::path(need("--outdir"));
            else {
                cerr << "unknown arg: " << a << "\n";
                cerr << "usage: " << argv[0] << " [--outdir DIR]\n";
                return 1;
            }
        }

        // ---- CKKS 参数 ----
        const size_t polyN = 8192;
        const vector<int> mod_bits = {60, 40, 40, 60};
        const double scale = pow(2.0, 40);

        EncryptionParameters parms(scheme_type::ckks);
        parms.set_poly_modulus_degree(polyN);
        parms.set_coeff_modulus(CoeffModulus::Create(polyN, mod_bits));
        SEALContext ctx(parms, /*expand_mod_chain=*/true);

        // ---- 保存 context ----
        {
            stringstream s; parms.save(s);
            save_bytes(outdir / "context.seal", s);
        }

        // ---- KeyGen ----
        KeyGenerator kg(ctx);
        const auto sk = kg.secret_key();

        PublicKey pk;  kg.create_public_key(pk);
        RelinKeys rlk; kg.create_relin_keys(rlk);
        GaloisKeys gk; kg.create_galois_keys(gk);  // 生成完整旋转集（SEAL 会按需裁剪存储）

        // ---- 写文件 ----
        {
            stringstream s; const_cast<SecretKey&>(sk).save(s);
            save_bytes(outdir / "sk.bin", s);
        }
        {
            stringstream s; pk.save(s);
            save_bytes(outdir / "pk.bin", s);
        }
        {
            stringstream s; rlk.save(s);
            save_bytes(outdir / "relin.bin", s);
        }
        {
            stringstream s; gk.save(s);
            save_bytes(outdir / "galois.bin", s);
        }

        // ---- 清单/提示 ----
        cout << "[ckks_keygen] Generated files in: " << outdir << "\n"
             << "  - context.seal\n"
             << "  - pk.bin\n"
             << "  - sk.bin\n"
             << "  - relin.bin\n"
             << "  - galois.bin\n"
             << "params: polyN=" << polyN
             << "  coeff_mod_bits=[" << mod_bits[0] << "," << mod_bits[1] << ","
             << mod_bits[2] << "," << mod_bits[3] << "]"
             << "  scale=2^40\n";
        return 0;
    } catch (const exception& e) {
        cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}

