// ckks_decrypt_dump.cpp — 解密 ./scores_raw 下密文，按实际 scale 归一；导出 CSV
#include <seal/seal.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <optional>
#include <memory>
#include <cmath>
#include <cstring>

using namespace std;
using namespace seal;
namespace fs = std::filesystem;

// ---------- 小工具 ----------
static vector<uint8_t> read_all(const fs::path& p) {
    ifstream f(p, ios::binary);
    if (!f.is_open()) throw runtime_error("open failed: " + p.string());
    f.seekg(0, ios::end);
    auto n = f.tellg();
    if (n < 0) throw runtime_error("tellg failed: " + p.string());
    f.seekg(0);
    vector<uint8_t> b((size_t)n);
    if (n > 0) f.read((char*)b.data(), (size_t)n);
    return b;
}

static void list_add_scores_raw(vector<fs::path>& out, const fs::path& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return;
    for (auto &e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        auto name = e.path().filename().string();
        if (name.rfind("scores_", 0) == 0 && e.path().extension() == ".bin")
            out.push_back(e.path());
    }
}

static void usage(const char* prog){
    cerr <<
    "Usage:\n"
    "  " << prog << " [--context context.seal] [--sk sk.bin]\n"
    "               [--scores_dir ./scores_raw] [--dim N]\n"
    "\n"
    "Notes:\n"
    "  - 不带 --dim：归一=scale（点积和）\n"
    "  - 带 --dim N：归一=scale*N（平均相似度）\n";
}

// ---------- 主程序 ----------
int main(int argc, char** argv) {
    try {
        // CLI
        fs::path ctx_p = fs::current_path() / "context.seal";
        fs::path sk_p  = fs::current_path() / "sk.bin";
        fs::path scores_dir = fs::current_path() / "scores_raw";
        long dim_arg = -1; // <0 表示未指定

        auto need = [&](int &i)->const char*{
            if (i+1 >= argc) { usage(argv[0]); std::exit(1); }
            return argv[++i];
        };
        for (int i=1; i<argc; ++i) {
            string a = argv[i];
            if      (a=="--context")    ctx_p      = need(i);
            else if (a=="--sk")         sk_p       = need(i);
            else if (a=="--scores_dir") scores_dir = need(i);
            else if (a=="--dim")        dim_arg    = std::stol(need(i));
            else { usage(argv[0]); return 1; }
        }

        if (!fs::exists(ctx_p)) { cerr << "missing: " << ctx_p << "\n"; return 1; }
        if (!fs::exists(sk_p))  { cerr << "missing: " << sk_p  << "\n"; return 1; }

        // 1) 还原参数与上下文（expand_mod_chain=true）
        EncryptionParameters parms;
        {
            auto buf = read_all(ctx_p);
            string s((const char*)buf.data(), buf.size());
            stringstream ss(s);
            parms.load(ss);
        }
        SEALContext ctx(parms, /*expand_mod_chain=*/true);

        cout << "poly_modulus_degree=" << parms.poly_modulus_degree()
             << "  coeff_modulus_size=" << parms.coeff_modulus().size()
             << "  bits=[";
        for (size_t i=0;i<parms.coeff_modulus().size();++i) {
            if (i) cout << ",";
            cout << parms.coeff_modulus()[i].bit_count();
        }
        cout << "]\n";

        // 2) SecretKey
        SecretKey sk;
        {
            auto buf = read_all(sk_p);
            string s((const char*)buf.data(), buf.size());
            stringstream ss(s);
            sk.load(ctx, ss);
        }
        Decryptor dec(ctx, sk);
        CKKSEncoder enc(ctx);

        // 3) 仅收集 scores_dir 下的密文
        vector<fs::path> files;
        list_add_scores_raw(files, scores_dir);
        sort(files.begin(), files.end());
        files.erase(unique(files.begin(), files.end()), files.end());

        if (files.empty()) {
            cerr << "no scores_*.bin found in " << scores_dir << "\n";
            return 1;
        }
        cout << "found " << files.size() << " files in " << scores_dir << "\n";

        // 4) 固定小数格式
        std::cout.setf(std::ios::fixed);
        std::cout.precision(10);

        auto try_load = [&](const SEALContext& c, const vector<uint8_t>& data) -> optional<Ciphertext> {
            try {
                string s((const char*)data.data(), data.size());
                stringstream ss(s);
                Ciphertext ct; ct.load(c, ss);
                return ct;
            } catch (...) {
                return {};
            }
        };

        size_t idx = 0;
        for (auto &p : files) {
            try {
                auto buf = read_all(p);

                // 先按原样尝试
                optional<Ciphertext> oct = try_load(ctx, buf);

                // 若失败，尝试去掉尾部 0（恒定化 padding）
                if (!oct) {
                    vector<uint8_t> tmp = buf;
                    while (!tmp.empty() && tmp.back() == 0) tmp.pop_back();
                    if (!tmp.empty()) oct = try_load(ctx, tmp);
                }

                if (!oct) {
                    cerr << "fail on " << p << " : ciphertext data is invalid (likely padded/trimmed)\n";
                    continue;
                }

                Ciphertext ct = *oct;

                // 打印 scale 和 level（chain_index）
                std::shared_ptr<const SEALContext::ContextData> cd = ctx.get_context_data(ct.parms_id());
                int level = cd ? static_cast<int>(cd->chain_index()) : -1;
                double scale_now = ct.scale();
                cout << "[meta] file=" << p.filename().string()
                     << "  scale=" << scale_now
                     << "  level=" << level << "\n";

                // 解密并 decode
                Plaintext pt; dec.decrypt(ct, pt);
                vector<double> v; enc.decode(pt, v);

                // 归一：未指定 dim -> 仅除 scale；指定 dim -> 除 (scale*dim)
                double norm = scale_now;
                if (dim_arg > 0) norm *= static_cast<double>(dim_arg);

                // 打印 preview（first8/mean）
                double mean = 0.0;
                for (double x : v) mean += x;
                if (!v.empty()) mean /= (double)v.size();

                cout << "[" << idx++ << "] " << p.filename().string()
                     << "  slots=" << v.size()
                     << "  first8=[";
                for (size_t i=0;i<min<size_t>(8, v.size()); ++i) {
                    if (i) cout << ", ";
                    cout << (v[i] / norm);
                }
                cout << "]  mean=" << (mean / norm) << "\n";

                // —— 导出 CSV（每个密文一个文件）——
                {
                    fs::path csvp = p;
                    csvp.replace_extension(".csv"); // scores_00.csv
                    std::ofstream of(csvp);
                    of.setf(std::ios::fixed); of.precision(10);
                    of << "slot,score\n";
                    for (size_t s = 0; s < v.size(); ++s) {
                        of << s << "," << (v[s] / norm) << "\n";
                    }
                    std::cout << "csv -> " << csvp << "\n";
                }

            } catch (const exception& e) {
                cerr << "fail on " << p << " : " << e.what() << "\n";
            }
        }
        return 0;
    } catch (const exception& e) {
        cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}

