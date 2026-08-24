#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "vector_store.hpp"
#include "meta_store.hpp"

// --------- 常量与枚举 ---------
namespace diag {
    constexpr uint32_t kMagic = 0xD1A6B10C;

    enum LayoutKind : uint32_t {
        ROW_MAJOR    = 1, // 旧版：count x dim 的行主矩阵
        OFFSET_MAJOR = 2  // 新版：多对角(BSGS)打包，按 offset 分块
    };
}

// --------- V2 版头部（覆盖 V1；字段向后兼容）---------
struct DiagBlockHeader {
    uint32_t magic;          // 固定 = diag::kMagic
    uint32_t version;        // 2 (多对角/offset-major)
    uint32_t dim;            // d
    uint32_t slots;          // pack_slots（CKKS 槽数，S_pad）
    uint32_t count;          // 实际候选数（<= slots）
    uint32_t stride;         // 目前=1（保留扩展位）
    uint64_t payload_bytes;  // 后随浮点 payload 的总字节数

    // V2 新增字段
    uint32_t num_offsets;    // diag_offsets 的长度（= 实际用到的偏移数）
    uint32_t layout_kind;    // diag::OFFSET_MAJOR(2) 或 diag::ROW_MAJOR(1)
    // 紧随其后是 payload：
    //  - 若 layout_kind == OFFSET_MAJOR:
    //      按 offset-major 排列：offset0 的 slots 个值，offset1 的 slots 个值，...（共 num_offsets 组）
    //  - 若 layout_kind == ROW_MAJOR:
    //      行主：count 行、每行 dim 个值（仅用于向后兼容，当前不再写这种）
};

// --------- 导出器接口 ---------
class DiagExporter {
public:
    DiagExporter(VectorStore* vs, const MetaInfo* meta, const std::string& data_dir);

    // 输入候选ID列表，按 pack_slots 切分成多个 block，返回布局
    void export_blocks(const std::vector<uint64_t>& cand_ids, uint32_t pack_slots,
                       std::vector<std::string>& block_ids,
                       std::vector<std::string>& mmap_paths,
                       std::vector<std::vector<uint32_t>>& diag_offsets_vec,
                       std::vector<uint32_t>& slots_vec,
                       std::vector<uint32_t>& stride_vec);

private:
    // 用 PQ codebook + code 重构近似向量 (float[d])
    bool reconstruct_vec(uint64_t cand_id, std::vector<float>& out);

    VectorStore* vs_;
    const MetaInfo* meta_;
    std::string data_dir_;
};

