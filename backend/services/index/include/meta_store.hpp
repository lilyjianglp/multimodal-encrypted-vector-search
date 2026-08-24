#pragma once
#include <sqlite3.h>
#include <string>
#include <vector>
#include <cstdint>

struct MetaInfo {
    uint32_t dim = 0;
    uint32_t K = 0;
    uint32_t M = 0;
    uint32_t Ks = 0;
    uint64_t N = 0;
    std::string centers_sha;
};

class MetaStore {
public:
    explicit MetaStore(const std::string& db_path);
    ~MetaStore();
    void init_schema();
    void write_meta(const MetaInfo& m);
    MetaInfo read_meta();
    // postings: (cluster_id → ordered candidate ids + approx_score)
    void insert_posting(uint32_t cluster_id, uint64_t cand_id, float approx_score);
    void get_topR(uint32_t cluster_id, uint32_t R,
                  std::vector<uint64_t>& ids, std::vector<float>& scores);
    void begin_tx();
    void commit_tx();

private:
    sqlite3* db_{nullptr};
};
