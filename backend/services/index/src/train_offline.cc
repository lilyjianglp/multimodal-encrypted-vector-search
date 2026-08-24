// train_offline.cc — 建索引（L2 归一化 + KMeans + PQ + postings）(with progress bars)
// 构建：g++/clang++ 或 CMake（需要 Faiss、OpenSSL、SQLite 等依赖）
//
// 用法：
//   ./train_offline <data_dir> N d K M Ks gen_random(0/1) [--fvecs=/path/base.fvecs] [--f32=/path/corpus.f32]
// 说明：
//   - 若提供 --fvecs，则按 fvecs 读取（每条前 4 字节为维度 d；覆盖命令行 N,d）
//   - 若提供 --f32，则按 float32 原始矩阵读取（形状 [N, d]），使用命令行 N,d
//   - 若都未提供，且 gen_random=1 则随机生成 N×d 的高斯向量（演示）
//   - 训练/编码均在“L2 归一化后”的空间上进行
//
// 输出：
//   <data_dir>/centers.snap        (K × d floats, 单位向量中心)
//   <data_dir>/pq_codebook.snap    (PQ 码本)
//   <data_dir>/pq_codes.snap       (N × M bytes, 每条向量的 M 个码字索引)
//   <data_dir>/index.db            (SQLite，含 meta 与 postings)

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/ProductQuantizer.h>

#include <random>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include "meta_store.hpp"   // 你现有的 SQLite 元数据/倒排写入工具
#include <openssl/sha.h>

// ---------- 简易进度条 ----------
struct Progress {
    std::string title;
    size_t total{0};
    size_t last_printed{0};
    int width{40};
    std::chrono::steady_clock::time_point start;

    Progress(std::string t, size_t tot, int w=40)
        : title(std::move(t)), total(tot), width(w), start(std::chrono::steady_clock::now()) {
        print(0);
    }
    void update(size_t done) {
        size_t step = std::max<size_t>(total / 100, 1);
        if (done == total || done - last_printed >= step) {
            last_printed = done;
            print(done);
        }
    }
    void finish() { print(total, /*done=*/true); std::cerr << "\n"; }

private:
    void print(size_t done, bool done_flag=false) {
        double ratio = total ? (double)done / (double)total : 1.0;
        ratio = std::clamp(ratio, 0.0, 1.0);
        int filled = (int)std::round(ratio * width);
        auto now = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(now - start).count();
        double eta  = ratio > 1e-9 ? secs * (1.0 / ratio - 1.0) : NAN;

        std::cerr << "\r" << title << " [";
        for (int i=0;i<width;i++) std::cerr << (i < filled ? '=' : ' ');
        std::cerr << "] " << std::fixed << std::setprecision(1) << (ratio*100.0) << "%";

        if (!done_flag) {
            if (std::isfinite(eta)) std::cerr << "  ETA: " << std::setprecision(1) << eta << "s";
            else std::cerr << "  ETA: --";
        } else {
            std::cerr << "  Elapsed: " << std::setprecision(1) << secs << "s";
        }
        std::cerr << std::flush;
    }
};

// ---------- L2 归一化（带进度） ----------
static inline void l2_normalize_rows(std::vector<float>& X, size_t N, int d) {
    const float eps = 1e-30f;
    const size_t block = 1u << 15; // 32768 行一刷
    Progress bar("L2 normalize", N);
    for (size_t i = 0; i < N; ++i) {
        float sum = 0.f;
        float* row = X.data() + i * (size_t)d;
        for (int j = 0; j < d; ++j) sum += row[j] * row[j];
        sum = std::sqrt(std::max(sum, eps));
        const float inv = 1.0f / sum;
        for (int j = 0; j < d; ++j) row[j] *= inv;
        if ((i % block) == 0 || i + 1 == N) bar.update(i + 1);
    }
    bar.finish();
}

// --- 读取 .fvecs：每条前 4 字节为维度 d，后接 d 个 float（带进度） ---
static bool load_fvecs(const std::string& path, std::vector<float>& out, size_t& n, int& d) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    int dd = 0; size_t cnt = 0;
    while (true) {
        if (fread(&dd, 4, 1, f) != 1) break;
        if (cnt == 0) d = dd;
        else if (dd != d) { fclose(f); return false; }
        if (fseek(f, (long)d * 4, SEEK_CUR) != 0) { fclose(f); return false; }
        cnt++;
    }
    if (cnt == 0) { fclose(f); return false; }

    out.resize(cnt * (size_t)d);
    n = cnt;

    rewind(f);
    Progress bar("Load .fvecs", cnt);
    for (size_t i = 0; i < cnt; ++i) {
        if (fread(&dd, 4, 1, f) != 1 || dd != d) { fclose(f); return false; }
        if (fread(out.data() + i * d, 4, d, f) != (size_t)d) { fclose(f); return false; }
        bar.update(i + 1);
    }
    bar.finish();
    fclose(f);
    return true;
}

// --- 读取 .f32：原始 float32，形状 [N, d]（分块 + 进度）---
static bool load_f32(const std::string& path, std::vector<float>& out, size_t N, int d) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    const size_t need = N * (size_t)d;
    out.resize(need);

    const size_t block_rows = 1u << 14; // 16384 行一批
    Progress bar("Load .f32", N);
    size_t rows_read = 0;
    while (rows_read < N) {
        size_t r = std::min(block_rows, N - rows_read);
        size_t got = fread(out.data() + rows_read * (size_t)d, sizeof(float), r * (size_t)d, f);
        if (got != r * (size_t)d) {
            fclose(f);
            std::cerr << "f32 size mismatch at row " << rows_read << ", expect "
                      << r * (size_t)d << " floats, got " << got << "\n";
            return false;
        }
        rows_read += r;
        bar.update(rows_read);
    }
    bar.finish();
    fclose(f);
    return true;
}

static void save_bin(const std::string& path, const void* buf, size_t bytes) {
    std::ofstream ofs(path, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf), bytes);
    ofs.close();
}

static std::string sha256_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) return "";
    SHA256_CTX ctx; SHA256_Init(&ctx);
    unsigned char buf[1<<16]; size_t n;
    while((n=fread(buf,1,sizeof(buf),f))>0) SHA256_Update(&ctx, buf, n);
    fclose(f);
    unsigned char out[32]; SHA256_Final(out, &ctx);
    static const char* hex="0123456789abcdef";
    std::string s(64,'0');
    for (int i=0;i<32;i++){ s[2*i]=hex[out[i]>>4]; s[2*i+1]=hex[out[i]&0xF]; }
    return s;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <data_dir> N d K M Ks gen_random(0/1) [--fvecs=/path/base.fvecs] [--f32=/path/corpus.f32]\n";
        return 1;
    }
    std::string data_dir = argv[1];
    size_t N = std::stoull(argv[2]);
    int d = std::stoi(argv[3]);
    int K = std::stoi(argv[4]);
    int M = std::stoi(argv[5]);
    int Ks = std::stoi(argv[6]); // 256 -> nbits=8
    bool gen_rand = std::stoi(argv[7]) != 0;

    // 解析可选参数
    std::string fvecs_path, f32_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        const std::string k1 = "--fvecs=";
        const std::string k2 = "--f32=";
        if (arg.rfind(k1, 0) == 0) fvecs_path = arg.substr(k1.size());
        if (arg.rfind(k2, 0) == 0) f32_path   = arg.substr(k2.size());
    }

    // 1) 准备数据 X: float32 [N][d]
    std::vector<float> X;

    if (!fvecs_path.empty()) {
        size_t N_loaded = 0; int d_loaded = d;
        if (!load_fvecs(fvecs_path, X, N_loaded, d_loaded)) {
            std::cerr << "Failed to load fvecs: " << fvecs_path << std::endl;
            return 3;
        }
        N = N_loaded; d = d_loaded;
        std::cerr << "Loaded fvecs N=" << N << " d=" << d << " from " << fvecs_path << std::endl;
    } else if (!f32_path.empty()) {
        if (!load_f32(f32_path, X, N, d)) {
            std::cerr << "Failed to load f32: " << f32_path << " (expect N*d floats)\n";
            return 3;
        }
        std::cerr << "Loaded f32 N=" << N << " d=" << d << " from " << f32_path << std::endl;
    } else if (gen_rand) {
        X.resize((size_t)N * d);
        std::mt19937 rng(42);
        std::normal_distribution<float> ga(0.0f, 1.0f);
        Progress bar("Generate random", N);
        for (size_t i=0;i<N;i++) {
            float* row = X.data() + i*(size_t)d;
            for (int j=0;j<d;j++) row[j] = ga(rng);
            bar.update(i+1);
        }
        bar.finish();
        std::cerr << "Generated random N=" << N << " d=" << d << std::endl;
    } else {
        std::cerr << "No data source: use --fvecs=... or --f32=... or set gen_random=1\n";
        return 2;
    }

    if (d % M != 0) {
        std::cerr << "Error: dim d=" << d << " must be divisible by M=" << M << std::endl;
        return 4;
    }

    // 2) 训练/编码之前做 L2 归一化
    l2_normalize_rows(X, N, d);

    // 3) 训练 KMeans 得 centers
    std::cerr << "\n[Stage] KMeans training (K=" << K << ", d=" << d << ")\n";
    faiss::Clustering clus(d, K);
    clus.verbose = true;          // 打开 faiss 自带日志
    faiss::IndexFlatL2 km_index(d);
    clus.train(N, X.data(), km_index);

    // 保存 centers.snap 与 SHA256
    std::string cpath = data_dir + "/centers.snap";
    save_bin(cpath, clus.centroids.data(), sizeof(float) * (size_t)K * d);
    std::string csha = sha256_file(cpath);
    std::cerr << "[OK] centers.snap saved. sha256=" << csha.substr(0, 16) << "...\n";

    // 4) 分配 cluster（1NN）—— 分批 search 以显示进度
    std::cerr << "\n[Stage] Assign points to nearest center (1NN)\n";
    faiss::IndexFlatL2 centers_index(d);
    centers_index.add(K, clus.centroids.data());

    std::vector<faiss::Index::idx_t> assign(N);
    std::vector<float> dists(N); // L2^2

    const size_t B = std::max<size_t>(32768, 1u<<15);
    Progress pa("Assign(1NN)", N);
    for (size_t off = 0; off < N; off += B) {
        size_t n_now = std::min(B, N - off);
        centers_index.search(n_now, X.data() + off*(size_t)d, 1,
                             dists.data() + off, assign.data() + off);
        pa.update(off + n_now);
    }
    pa.finish();

    // 5) 训练 PQ
    std::cerr << "\n[Stage] PQ training (M=" << M << ", Ks=" << Ks << ")\n";
    int nbits = (int)std::round(std::log2((double)Ks));
    faiss::ProductQuantizer pq(d, M, nbits);
    pq.verbose = true;            // 打印 PQ 训练日志
    pq.train(N, X.data());

    // 导出 PQ 码本
    save_bin(data_dir + "/pq_codebook.snap", pq.centroids.data(),
             pq.centroids.size() * sizeof(float));
    std::cerr << "[OK] pq_codebook.snap saved.\n";

    // 6) 编码所有向量 -> pq_codes.snap（N*M bytes）(分批)
    std::cerr << "\n[Stage] PQ compute codes & save (N*M bytes)\n";
    std::vector<uint8_t> codes((size_t)N * M);
    Progress pc("PQ encode", N);
    for (size_t off = 0; off < N; off += B) {
        size_t n_now = std::min(B, N - off);
        pq.compute_codes(X.data() + off*(size_t)d, codes.data() + off*(size_t)M, n_now);
        pc.update(off + n_now);
    }
    pc.finish();
    save_bin(data_dir + "/pq_codes.snap", codes.data(), codes.size());
    std::cerr << "[OK] pq_codes.snap saved. Bytes=" << codes.size() << "\n";

    // 7) 写 SQLite 元数据 + postings（以“近似余弦分数”存）
    std::cerr << "\n[Stage] Write SQLite meta & postings\n";
    MetaStore ms(data_dir + "/index.db");
    ms.init_schema();

    MetaInfo mi;
    mi.dim = d; mi.K = K; mi.M = M; mi.Ks = Ks; mi.N = N;
    mi.centers_sha = csha;
    ms.write_meta(mi);

    ms.begin_tx();
    Progress ps("SQLite postings", N);
    for (size_t off = 0; off < N; off += B) {
        size_t n_now = std::min(B, N - off);
        for (size_t i = 0; i < n_now; ++i) {
            size_t idx = off + i;
            float cos_approx = 1.0f - 0.5f * dists[idx];
            ms.insert_posting((uint32_t)assign[idx], (uint64_t)idx, cos_approx);
        }
        ps.update(off + n_now);
    }
    ms.commit_tx();
    ps.finish();
    std::cerr << "[OK] index.db written.\n";

    std::cout << "Done. Wrote centers.snap / pq_codebook.snap / pq_codes.snap / index.db\n";
    return 0;
}

