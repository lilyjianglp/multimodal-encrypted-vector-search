#include "vector_store.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>

static void* map_file_ro(const std::string& p, size_t& out_bytes, int& out_fd) {
    out_fd = open(p.c_str(), O_RDONLY);
    if (out_fd<0) throw std::runtime_error("open failed: "+p);
    struct stat st; if (fstat(out_fd,&st)!=0) throw std::runtime_error("stat failed: "+p);
    out_bytes = (size_t)st.st_size;
    void* m = mmap(nullptr, out_bytes, PROT_READ, MAP_SHARED, out_fd, 0);
    if (m==MAP_FAILED) throw std::runtime_error("mmap failed: "+p);
    return m;
}

VectorStore::VectorStore(const std::string& data_dir): data_dir_(data_dir) {}

void VectorStore::load_snapshots() {
    // centers
    map_centers_ = map_file_ro(data_dir_+"/centers.snap", centers_nbytes_, fd_centers_);
    centers_ = reinterpret_cast<const float*>(map_centers_);
    // pq codes
    map_codes_ = map_file_ro(data_dir_+"/pq_codes.snap", pq_codes_nbytes_, fd_codes_);
    pq_codes_ = reinterpret_cast<const uint8_t*>(map_codes_);
    // codebook（可选）
    map_codebook_ = map_file_ro(data_dir_+"/pq_codebook.snap", pq_codebook_nbytes_, fd_codebook_);
    pq_codebook_ = reinterpret_cast<const float*>(map_codebook_);
}

bool VectorStore::read_pq_code(uint64_t cand_id, uint32_t M, std::string& out_code) const {
    size_t off = cand_id * (size_t)M;
    if ((off+M) > pq_codes_nbytes_) return false;
    out_code.resize(M);
    std::memcpy(out_code.data(), pq_codes_ + off, M);
    return true;
}
