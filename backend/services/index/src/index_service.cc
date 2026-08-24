#include <nlohmann/json.hpp>
using json = nlohmann::json;
#include "index_service.hpp"
#include <grpcpp/grpcpp.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include "vector_store.hpp"
#include "meta_store.hpp"
#include "diag_export.hpp"
#include <iostream>
// index_service.cc 顶部（#include 之后，文件内任意位置，建议匿名命名空间）
namespace {
inline uint32_t ceil_pow2(uint32_t x) {               // 向上取 2 的幂
    if (x <= 1) return 1;
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16;
    return x + 1;
}
inline uint32_t clamp_pow2(uint32_t v,
                           uint32_t defv = 1024,     // 和 B 的 CKKS 槽位约好
                           uint32_t minv = 8,
                           uint32_t maxv = 16384) {
    if (v == 0) v = defv;
    if (v < minv) v = minv;
    if (v > maxv) v = maxv;
    uint32_t p = ceil_pow2(v);                       // 强制对齐到 2 的幂
    if (p > maxv) p = maxv;
    return p;
}
} // namespace

static std::unique_ptr<VectorStore> g_vs;
static std::unique_ptr<MetaStore> g_ms;
static MetaInfo g_meta;

static std::unique_ptr<DiagExporter> g_dx;

IndexServiceImpl::IndexServiceImpl(std::string data_dir)
: data_dir_(std::move(data_dir)) {
  if (!std::filesystem::exists(data_dir_)) std::filesystem::create_directories(data_dir_);
  g_vs = std::make_unique<VectorStore>(data_dir_);
  g_vs->load_snapshots();
  g_ms = std::make_unique<MetaStore>(data_dir_ + "/index.db");
  g_meta = g_ms->read_meta();
  if (g_meta.dim == 0 || g_meta.M == 0 || g_meta.Ks == 0 || (g_meta.dim % g_meta.M != 0)) {
     std::cerr << "[IndexService] invalid meta: d=" << g_meta.dim
               << " M=" << g_meta.M << " Ks=" << g_meta.Ks << std::endl;
     // 这里不直接 return，让服务启动失败更明显（或抛异常）
     throw std::runtime_error("invalid meta, check snapshots and index.db");
 }
  g_dx = std::make_unique<DiagExporter>(g_vs.get(), &g_meta, data_dir_);
}

::grpc::Status IndexServiceImpl::GetCenters(::grpc::ServerContext*,
    const indexsvc::GetCentersRequest*, indexsvc::GetCentersResponse* resp) {
    resp->set_centers_sha(g_meta.centers_sha);
    resp->set_centers_path(g_vs->centers_path());
    resp->set_dim(g_meta.dim);
    resp->set_num_centers(g_meta.K);
    return ::grpc::Status::OK;
}


::grpc::Status IndexServiceImpl::GetClusterCandidates(::grpc::ServerContext*,
    const indexsvc::GetClusterCandidatesRequest* req,
    indexsvc::GetClusterCandidatesResponse* resp) {

    std::vector<uint64_t> ids; std::vector<float> scores;
    g_ms->get_topR(req->cluster_id(), req->top_r(), ids, scores);

    // 不足补齐 dummy
    while (ids.size() < req->top_r()) { ids.push_back(0); scores.push_back(0.0f); }

    for (size_t i = 0; i < ids.size(); ++i) {
        auto* c = resp->add_candidates();
        c->set_id(ids[i]);
        c->set_approx_score(scores[i]);
        std::string code;
        if (!g_vs->read_pq_code(ids[i], g_meta.M, code)) {
            code.assign(g_meta.M, 0);
        }
        c->set_pq_code(code);
    }
    resp->set_padded_to(req->top_r());
    return ::grpc::Status::OK;
}

::grpc::Status IndexServiceImpl::CreateDiagBlocks(
    ::grpc::ServerContext*,
    const indexsvc::CreateDiagBlocksRequest* req,
    indexsvc::CreateDiagBlocksResponse* resp)
{
    namespace fs = std::filesystem;

    // ---- 0) 读取候选 ID ----
    std::vector<uint64_t> cand_ids(req->candidate_ids().begin(),
                                   req->candidate_ids().end());
    if (cand_ids.empty()) {
        return ::grpc::Status::OK;
    }

    // ---- 1) diag_blocks 根目录 ----
    fs::path root = fs::path(data_dir_) / "diag_blocks";
    if (!fs::exists(root)) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                              "diag_blocks directory not found");
    }

    // ---- 2) 扫描所有 .dia 文件 ----
    std::vector<fs::path> dia_files;
    for (auto& p : fs::directory_iterator(root)) {
        if (p.path().extension() == ".dia") {
            dia_files.push_back(p.path());
        }
    }
    if (dia_files.empty()) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                              "no .dia files found in diag_blocks");
    }

    // 按 blk-xxxxxx.dia 排序
    std::sort(dia_files.begin(), dia_files.end(),
              [](const fs::path& a, const fs::path& b) {
                  return a.filename().string() < b.filename().string();
              });

    // ---- 3) 读取 diag_blocks.json ----
    fs::path meta_json = root / "diag_blocks.json";
    if (!fs::exists(meta_json)) {
        return ::grpc::Status(::grpc::StatusCode::NOT_FOUND,
                              "diag_blocks.json missing");
    }

    nlohmann::json jmeta;
    {
        std::ifstream fin(meta_json);
        fin >> jmeta;
    }

    auto blocks_j = jmeta["blocks"];
    if (!blocks_j.is_array()) {
        return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                              "diag_blocks.json: blocks not array");
    }

    if (blocks_j.size() != dia_files.size()) {
        return ::grpc::Status(::grpc::StatusCode::INTERNAL,
                              "mismatch: json blocks size != .dia files count");
    }

    // ---- 4) 构建返回 block 数据 + 正确生成 slot_ids ----
    size_t global_slot = 0;  // 全局槽指针（最终会生成 16384 长度的 slot_ids）

    for (size_t i = 0; i < dia_files.size(); ++i) {

        const auto& blk_json = blocks_j[i];
        const auto& fn       = dia_files[i].filename().string();

        auto* blk = resp->add_blocks();
        blk->set_block_id(fn);
        blk->set_mmap_path(fn);

        uint32_t slots  = blk_json["layout"]["slots"].get<uint32_t>();
        uint32_t stride = blk_json["layout"]["stride"].get<uint32_t>();
        blk->set_slots(slots);
        blk->set_stride(stride);

        // diag_offsets
        for (auto& v : blk_json["layout"]["diag_offsets"]) {
            blk->add_diag_offsets(v.get<uint32_t>());
        }

        // ---- ★★★ 正确为每个 block 生成 slot_ids ★★★
        for (size_t s = 0; s < slots; ++s, ++global_slot) {
            if (global_slot < cand_ids.size())
                blk->add_slot_ids(cand_ids[global_slot]);
            else
                blk->add_slot_ids(0);
        }
    }

    // ---- 5) 日志 ----
    std::cerr
        << "[CreateDiagBlocks] cand=" << cand_ids.size()
        << " slots=" << blocks_j[0]["layout"]["slots"]
        << " blocks=" << dia_files.size()
        << " first_path=" << dia_files[0].string()
        << std::endl;

    return ::grpc::Status::OK;
}





