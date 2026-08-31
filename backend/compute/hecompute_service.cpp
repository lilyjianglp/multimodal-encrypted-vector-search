#include "hecompute_service.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <seal/version.h>
#include <utility>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <future>
#include <thread>
#include <algorithm>
#include <iostream>
#include <map>

using namespace seal;

namespace {

static int getenv_i(const char* k, int d) {
if (const char* v = std::getenv(k)) {
return std::atoi(v);
}
return d;
}

static int InferGStep(const BSGSPlan& bsgs) {
if (bsgs.giant.size() >= 2) {
return std::abs(bsgs.giant[1] - bsgs.giant[0]);
}
return 0;
}

static std::pair<int, int> DecomposeRot(int rot, int g_step) {
int g = 0;
int b = rot;

if (g_step > 0) {
    if (rot >= 0) {
        g = (rot / g_step) * g_step;
    } else {
        g = -(((-rot) / g_step) * g_step);
    }
    b = rot - g;
}

return {g, b};

}

static bool IsFullBsgsPacking(const std::string& packing) {
return packing.find("bsgs-v1") != std::string::npos;
}

} // namespace

void HeComputeService::SetNumThreads(int n) {
#if defined(SEAL_VERSION_MAJOR) && (SEAL_VERSION_MAJOR < 4)
// 仅 SEAL 3.x 支持全局线程数控制
seal::util::global_variables::set_num_threads(n);
#else
// SEAL 4.x 不再支持该接口；真正并行需要在业务层做 block/batch 并行
(void)n;
#endif
}

void HeComputeService::EvalKeys(const std::string& client_id,
const std::string& key_ver,
const GaloisKeys& gal,
const RelinKeys& relin)
{
keycache_[CacheKey(client_id, key_ver)] = KeyBundle{gal, relin};
}

ScoreBatchReply HeComputeService::ScoreBatch(const ScoreBatchRequest& req)
{
auto it = keycache_.find(CacheKey(req.client_id, req.key_ver));
if (it == keycache_.end()) {
throw std::invalid_argument("E_MISSING_ROT_KEYS");
}

const auto& gal = it->second.gal;

Telemetry telem;
auto t0 = std::chrono::high_resolution_clock::now();

ScoreBatchReply reply;
reply.snapshot_id = req.blocks.empty() ? "" : req.blocks.front().snapshot_id;

if (req.blocks.empty()) {
    auto t1 = std::chrono::high_resolution_clock::now();
    telem.lat_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    reply.telemetry = telem;
    return reply;
}

const int g_step = InferGStep(req.bsgs);

/*
 * 关键优化 1：
 * baby rotation 不再每个 block 单独算，而是在整个 ScoreBatch 请求级别预计算一次。
 * 对 Top100 拆成 13 个 block 的情况，这一步可以减少大量重复 baby rotation。
 */
std::unordered_set<int> needed_babies;
needed_babies.reserve(128);

for (int b : req.bsgs.baby) {
    needed_babies.insert(b);
}

for (const auto& blk : req.blocks) {
    if (blk.layout.diag_offsets.size() != blk.diag_plaintexts.size()) {
        throw std::invalid_argument("diag_offsets size != diag_plaintexts size");
    }

    for (int rot : blk.layout.diag_offsets) {
        auto [g, b] = DecomposeRot(rot, g_step);
        (void)g;
        needed_babies.insert(b);
    }
}

std::unordered_map<int, Ciphertext> baby_cache;
baby_cache.reserve(needed_babies.size() + 8);

Evaluator pre_eval(context_);

for (int b : needed_babies) {
    Ciphertext tmp;
    if (b == 0) {
        tmp = req.ct_q;
    } else {
        pre_eval.rotate_vector(req.ct_q, b, gal, tmp);
        telem.rot_cnt++;
    }
    baby_cache.emplace(b, std::move(tmp));
}

const std::size_t n_blocks = req.blocks.size();

std::vector<Ciphertext> out_cts(n_blocks);
std::vector<std::array<std::size_t,2>> out_shapes(n_blocks);

/*
 * 关键优化 2：
 * 支持 block 级并行。
 * 默认 HE_BLOCK_PARALLEL=1，保持最稳。
 * 如果机器核心和内存足够，可以设置：
 *   export HE_BLOCK_PARALLEL=2
 *   export HE_BLOCK_PARALLEL=4
 */
int parallel = getenv_i("HE_BLOCK_PARALLEL", 1);
if (parallel < 1) parallel = 1;
parallel = std::min<int>(parallel, static_cast<int>(n_blocks));

auto compute_one = [&](std::size_t idx) -> Telemetry {
    Telemetry local_telem;
    Evaluator evaluator(context_);

    const auto& blk = req.blocks[idx];

    Ciphertext ct_scores = DotDiag_BSGS_Cached(
        req.ct_q,
        blk,
        req.bsgs,
        gal,
        evaluator,
        req.scale,
        baby_cache,
        g_step,
        local_telem
    );

    ct_scores.scale() = req.scale;

    out_cts[idx] = std::move(ct_scores);
    out_shapes[idx] = {1, blk.layout.slots};

    return local_telem;
};

if (parallel == 1 || n_blocks == 1) {
    for (std::size_t i = 0; i < n_blocks; ++i) {
        Telemetry lt = compute_one(i);
        telem.rot_cnt += lt.rot_cnt;
        telem.mul_cnt += lt.mul_cnt;
    }
} else {
    std::vector<std::future<Telemetry>> futs;
    futs.reserve(n_blocks);

    /*
     * 简单并行：每个 block 一个 async。
     * 如果 block 很多、机器内存紧张，可以后续改成固定线程池。
     */
    for (std::size_t i = 0; i < n_blocks; ++i) {
        futs.emplace_back(std::async(std::launch::async, compute_one, i));
    }

    for (auto& f : futs) {
        Telemetry lt = f.get();
        telem.rot_cnt += lt.rot_cnt;
        telem.mul_cnt += lt.mul_cnt;
    }
}

for (std::size_t i = 0; i < n_blocks; ++i) {
    reply.scores_ciphertexts.push_back(std::move(out_cts[i]));
    reply.pack_shapes.push_back(out_shapes[i]);
}

auto t1 = std::chrono::high_resolution_clock::now();
telem.lat_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
reply.telemetry = telem;

return reply;

}

Ciphertext HeComputeService::DotDiag_BSGS_Cached(
const Ciphertext& ct_q,
const DiagBlock& blk,
const BSGSPlan& bsgs,
const GaloisKeys& gal_keys,
Evaluator& evaluator,
double target_scale,
const std::unordered_map<int, Ciphertext>& baby_cache,
int g_step,
Telemetry& telem)
{
(void)ct_q;
(void)bsgs;

if (blk.layout.diag_offsets.size() != blk.diag_plaintexts.size()) {
    throw std::invalid_argument("diag_offsets size != diag_plaintexts size");
}

bool first = true;
Ciphertext acc;

/*
 * Full group-level BSGS layout.
 *
 * The offline generator stores, for r = g + b:
 *
 *   P'_{g,b} = RotPlain(P_r, -g)
 *
 * Hence an entire giant group can be accumulated before one ciphertext
 * rotation:
 *
 *   sum_r P_r * Rot(q, r)
 *     = sum_g Rot(sum_b P'_{g,b} * Rot(q, b), g)
 *
 * Online ciphertext rotations become O(sqrt(d)); plaintext multiplications
 * remain O(d). Legacy offset-major blocks continue through the path below.
 */
if (IsFullBsgsPacking(blk.layout.packing)) {
    if (g_step <= 0) {
        throw std::invalid_argument("full BSGS packing requires a positive giant step");
    }

    std::map<int, Ciphertext> group_sums;
    for (std::size_t i = 0; i < blk.layout.diag_offsets.size(); ++i) {
        const int rot = blk.layout.diag_offsets[i];
        auto [g, b] = DecomposeRot(rot, g_step);

        auto bit = baby_cache.find(b);
        if (bit == baby_cache.end()) {
            throw std::invalid_argument("missing baby rotation in cache");
        }

        Ciphertext prod;
        evaluator.multiply_plain(bit->second, blk.diag_plaintexts[i], prod);
        telem.mul_cnt++;

        auto group_it = group_sums.find(g);
        if (group_it == group_sums.end()) {
            group_sums.emplace(g, std::move(prod));
        } else {
            if (group_it->second.parms_id() != prod.parms_id()) {
                evaluator.mod_switch_to_inplace(group_it->second, prod.parms_id());
            }
            evaluator.add_inplace(group_it->second, prod);
        }
    }

    for (auto& [g, group_sum] : group_sums) {
        Ciphertext aligned;
        Ciphertext* term = &group_sum;
        if (g != 0) {
            evaluator.rotate_vector(group_sum, g, gal_keys, aligned);
            telem.rot_cnt++;
            term = &aligned;
        }

        if (first) {
            acc = std::move(*term);
            first = false;
        } else {
            if (acc.parms_id() != term->parms_id()) {
                evaluator.mod_switch_to_inplace(acc, term->parms_id());
            }
            evaluator.add_inplace(acc, *term);
        }
    }

    if (first) {
        throw std::invalid_argument("empty full BSGS diagonal block");
    }

    acc.scale() = target_scale;
    return acc;
}

for (std::size_t i = 0; i < blk.layout.diag_offsets.size(); ++i) {
    int rot = blk.layout.diag_offsets[i];

    auto [g, b] = DecomposeRot(rot, g_step);

    auto bit = baby_cache.find(b);
    if (bit == baby_cache.end()) {
        throw std::invalid_argument("missing baby rotation in cache");
    }

    const Ciphertext& baby_ct = bit->second;

    Ciphertext rotated;
    const Ciphertext* src = &baby_ct;

    /*
     * 保持旧版语义：
     * 旧版是先 baby rotate，再 giant rotate，再 multiply_plain。
     * 这里仍然保持这个顺序，避免改变 plaintext diagonal 对齐方式。
     */
    if (g != 0) {
        evaluator.rotate_vector(baby_ct, g, gal_keys, rotated);
        telem.rot_cnt++;
        src = &rotated;
    }

    Ciphertext prod;
    evaluator.multiply_plain(*src, blk.diag_plaintexts[i], prod);
    telem.mul_cnt++;

    if (first) {
        acc = std::move(prod);
        first = false;
    } else {
        if (acc.parms_id() != prod.parms_id()) {
            evaluator.mod_switch_to_inplace(acc, prod.parms_id());
        }
        evaluator.add_inplace(acc, prod);
    }
}

if (first) {
    throw std::invalid_argument("empty diag block");
}

acc.scale() = target_scale;
return acc;

}
