#include "diag_export.hpp"
#include <filesystem>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <cmath>

using std::uint32_t;
using std::uint64_t;

static inline std::string make_block_id(size_t idx) {
    char buf[32]; std::snprintf(buf, sizeof(buf), "blk-%06zu", idx);
    return std::string(buf);
}

DiagExporter::DiagExporter(VectorStore* vs, const MetaInfo* meta, const std::string& data_dir)
: vs_(vs), meta_(meta), data_dir_(data_dir) {}

bool DiagExporter::reconstruct_vec(uint64_t cand_id, std::vector<float>& out) {
    const uint32_t d = meta_->dim;
    const uint32_t M = meta_->M;
    const uint32_t Ks = meta_->Ks;
    const uint32_t dsub = d / M;

    std::string code;
    if (!vs_->read_pq_code(cand_id, M, code)) return false;

    out.resize(d);
    const float* cb = vs_->pq_codebook_data(); // [M][Ks][dsub]
    if (!cb) return false;

    for (uint32_t m = 0; m < M; ++m) {
        uint8_t c = static_cast<uint8_t>(code[m]);
        if (c >= Ks) c = 0;
        const float* centroid = cb + (size_t)m * Ks * dsub + (size_t)c * dsub;
        std::memcpy(out.data() + (size_t)m * dsub, centroid, sizeof(float)*dsub);
    }
    return true;
}

void DiagExporter::export_blocks(const std::vector<uint64_t>& cand_ids, uint32_t pack_slots,
                       std::vector<std::string>& block_ids,
                       std::vector<std::string>& mmap_paths,
                       std::vector<std::vector<uint32_t>>& diag_offsets_vec,
                       std::vector<uint32_t>& slots_vec,
                       std::vector<uint32_t>& stride_vec) {

    const uint32_t d = meta_->dim;
    const uint32_t M = meta_->M;
    const uint32_t Ks = meta_->Ks;
    if (d == 0 || M == 0 || Ks == 0) throw std::runtime_error("DiagExporter: invalid meta");
    if (d % M != 0) throw std::runtime_error("DiagExporter: dim must be divisible by M");

    std::filesystem::create_directories(data_dir_ + "/diag_blocks");

    // ---- BSGS 参数 ----
    const uint32_t b = static_cast<uint32_t>(std::ceil(std::sqrt((double)d)));
    const uint32_t g = (d + b - 1) / b; // ceil(d/b)

    // 生成 offsets：u*b+v（过滤 >= d）
    std::vector<uint32_t> all_offsets;
    all_offsets.reserve(d);
    for (uint32_t u = 0; u < g; ++u) {
        for (uint32_t v = 0; v < b; ++v) {
            uint32_t t = u * b + v;
            if (t < d) all_offsets.push_back(t);
        }
    }
    const uint32_t num_offsets = (uint32_t)all_offsets.size();

    const size_t total = cand_ids.size();
    const size_t num_blocks = (total + pack_slots - 1) / pack_slots;

    block_ids.reserve(num_blocks);
    mmap_paths.reserve(num_blocks);
    diag_offsets_vec.reserve(num_blocks);
    slots_vec.reserve(num_blocks);
    stride_vec.reserve(num_blocks);

    for (size_t bidx = 0; bidx < num_blocks; ++bidx) {
        const size_t begin = bidx * pack_slots;
        const size_t end   = std::min(begin + pack_slots, total);
        const uint32_t S   = static_cast<uint32_t>(end - begin);  // 实际候选数
        const uint32_t S_pad = pack_slots;                        // 固定槽位数

        const std::string bid  = make_block_id(bidx);
        const std::string path = data_dir_ + "/diag_blocks/" + bid + ".bin";

        // 打开输出文件
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) throw std::runtime_error("open mmap path failed: " + path);

        // 写 Header（V2，OFFSET_MAJOR）
        DiagBlockHeader hdr{};
        hdr.magic         = diag::kMagic;
        hdr.version       = 2;
        hdr.dim           = d;
        hdr.slots         = S_pad;
        hdr.count         = S;
        hdr.stride        = 1;
        hdr.payload_bytes = sizeof(float) * (size_t)num_offsets * S_pad;
        hdr.num_offsets   = num_offsets;
        hdr.layout_kind   = diag::OFFSET_MAJOR;

        static_assert(sizeof(DiagBlockHeader) == 40, "DiagBlockHeader size must be 40 bytes");
        ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

        // 先重构 S 个候选的完整向量（大小 S x d）
        std::vector<std::vector<float>> rows(S);
        std::vector<float> recon;
        for (uint32_t i = 0; i < S; ++i) {
            if (!reconstruct_vec(cand_ids[begin + i], recon)) {
                rows[i].assign(d, 0.f);
            } else {
                rows[i] = std::move(recon);
            }
        }

        // offset-major 写 payload：每个 offset 一条长度 slots 的向量
        std::vector<float> slotvec(S_pad, 0.f);
        for (uint32_t off : all_offsets) {
            // The encrypted query is repeated every d slots. A rotation by
            // `off` puts q[(s + off) % d] at output slot s, so use the same
            // candidate coordinate on this diagonal.
            for (uint32_t s = 0; s < S; ++s) {
                slotvec[s] = rows[s][((size_t)s + off) % d];
            }
            for (uint32_t s = S; s < S_pad; ++s) slotvec[s] = 0.f;
            ofs.write(reinterpret_cast<const char*>(slotvec.data()), sizeof(float)*S_pad);
        }
        ofs.close();

        // 输出给上层
        block_ids.push_back(bid);
        mmap_paths.push_back(path);
        diag_offsets_vec.push_back(all_offsets);
        slots_vec.push_back(S_pad);
        stride_vec.push_back(1);
    }
}
