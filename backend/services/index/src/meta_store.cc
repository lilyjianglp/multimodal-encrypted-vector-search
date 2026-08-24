#include "meta_store.hpp"
#include <stdexcept>
#include <cstring>

static void check(int rc, sqlite3* db, const char* msg) {
    if (rc != SQLITE_OK && rc != SQLITE_DONE && rc != SQLITE_ROW) {
        std::string e = msg; e += ": "; e += sqlite3_errmsg(db);
        throw std::runtime_error(e);
    }
}

MetaStore::MetaStore(const std::string& db_path) {
    if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK)
        throw std::runtime_error("open sqlite failed");
}
MetaStore::~MetaStore() { if (db_) sqlite3_close(db_); }

void MetaStore::init_schema() {
    const char* ddl =
        "PRAGMA journal_mode=WAL;"  // 写入并发优化
        "CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);"
        "CREATE TABLE IF NOT EXISTS postings ("
        " cluster_id INTEGER, cand_id INTEGER, approx_score REAL,"
        " PRIMARY KEY(cluster_id, cand_id)"
        ");"
        "CREATE INDEX IF NOT EXISTS idx_postings_cluster_score "
        " ON postings(cluster_id, approx_score DESC);";  // 新增索引

    char* err = nullptr;
    int rc = sqlite3_exec(db_, ddl, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::string e = err ? err : "";
        sqlite3_free(err);
        throw std::runtime_error(e);
    }
}


void MetaStore::write_meta(const MetaInfo& m) {
    auto put = [&](const char* k, const std::string& v){
        sqlite3_stmt* st=nullptr;
        check(sqlite3_prepare_v2(db_, "REPLACE INTO meta(k,v) VALUES(?,?)", -1, &st, nullptr), db_, "prep");
        check(sqlite3_bind_text(st,1,k,-1,SQLITE_STATIC), db_, "bind1");
        check(sqlite3_bind_text(st,2,v.c_str(),-1,SQLITE_STATIC), db_, "bind2");
        check(sqlite3_step(st), db_, "step");
        sqlite3_finalize(st);
    };
    put("dim", std::to_string(m.dim));
    put("K", std::to_string(m.K));
    put("M", std::to_string(m.M));
    put("Ks", std::to_string(m.Ks));
    put("N", std::to_string(m.N));
    put("centers_sha", m.centers_sha);
}

MetaInfo MetaStore::read_meta() {
    MetaInfo m;
    const char* q="SELECT k,v FROM meta";
    sqlite3_stmt* st=nullptr;
    check(sqlite3_prepare_v2(db_, q, -1, &st, nullptr), db_, "prep");
    while (sqlite3_step(st)==SQLITE_ROW) {
        std::string k=(const char*)sqlite3_column_text(st,0);
        std::string v=(const char*)sqlite3_column_text(st,1);
        if(k=="dim") m.dim=std::stoul(v);
        else if(k=="K") m.K=std::stoul(v);
        else if(k=="M") m.M=std::stoul(v);
        else if(k=="Ks") m.Ks=std::stoul(v);
        else if(k=="N") m.N=std::stoull(v);
        else if(k=="centers_sha") m.centers_sha=v;
    }
    sqlite3_finalize(st);
    return m;
}

void MetaStore::insert_posting(uint32_t cluster_id, uint64_t cand_id, float approx_score) {
    sqlite3_stmt* st=nullptr;
    check(sqlite3_prepare_v2(db_,
        "REPLACE INTO postings(cluster_id,cand_id,approx_score) VALUES(?,?,?)", -1, &st, nullptr), db_, "prep");
    check(sqlite3_bind_int(st,1,(int)cluster_id), db_, "b1");
    check(sqlite3_bind_int64(st,2,(sqlite3_int64)cand_id), db_, "b2");
    check(sqlite3_bind_double(st,3,(double)approx_score), db_, "b3");
    check(sqlite3_step(st), db_, "step");
    sqlite3_finalize(st);
}

void MetaStore::get_topR(uint32_t cluster_id, uint32_t R,
                  std::vector<uint64_t>& ids, std::vector<float>& scores) {
    sqlite3_stmt* st=nullptr;
    check(sqlite3_prepare_v2(db_,
        "SELECT cand_id, approx_score FROM postings WHERE cluster_id=? ORDER BY approx_score DESC LIMIT ?", -1, &st, nullptr),
        db_, "prep");
    check(sqlite3_bind_int(st,1,(int)cluster_id), db_, "b1");
    check(sqlite3_bind_int(st,2,(int)R), db_, "b2");
    while (sqlite3_step(st)==SQLITE_ROW) {
        ids.push_back((uint64_t)sqlite3_column_int64(st,0));
        scores.push_back((float)sqlite3_column_double(st,1));
    }
    sqlite3_finalize(st);
}
void MetaStore::begin_tx() {
    char* err = nullptr;
    int rc = sqlite3_exec(db_, "BEGIN IMMEDIATE;", nullptr, nullptr, &err);
    if (rc != SQLITE_OK) { std::string e = err ? err : ""; sqlite3_free(err); throw std::runtime_error("BEGIN IMMEDIATE failed: " + e); }
}

void MetaStore::commit_tx() {
    char* err = nullptr;
    int rc = sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &err);
    if (rc != SQLITE_OK) { std::string e = err ? err : ""; sqlite3_free(err); throw std::runtime_error("COMMIT failed: " + e); }
}

