#pragma once
#include <string>
#include <vector>
#include <cstdint>

class VectorStore {
public:
    VectorStore(const std::string& data_dir);
    // memory-map snapshots
    void load_snapshots();

    // centers
    const float* centers_data() const { return centers_; }
    size_t centers_bytes() const { return centers_nbytes_; }

    // pq codes
    bool read_pq_code(uint64_t cand_id, uint32_t M, std::string& out_code) const;

    // pq codebook（可选：需要导出对角块时用）
    const float* pq_codebook_data() const { return pq_codebook_; }
    size_t pq_codebook_bytes() const { return pq_codebook_nbytes_; }

    std::string centers_path() const { return data_dir_ + "/centers.snap"; }

private:
    std::string data_dir_;
    // mmap pointers
    int fd_centers_=-1, fd_codes_=-1, fd_codebook_=-1;
    void* map_centers_=nullptr; size_t centers_nbytes_=0;
    void* map_codes_=nullptr; size_t pq_codes_nbytes_=0;
    void* map_codebook_=nullptr; size_t pq_codebook_nbytes_=0;
    const float* centers_=nullptr;
    const uint8_t* pq_codes_=nullptr;
    const float* pq_codebook_=nullptr;
};
