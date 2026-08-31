#include "../compute/hecompute_service.hpp"

#include <seal/seal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace seal;

namespace {

struct Fixture {
    std::uint32_t query_count{};
    std::uint32_t candidate_count{};
    std::uint32_t dimension{};
    std::uint32_t topk{};
    std::vector<std::int64_t> candidate_ids;
    std::vector<std::int64_t> exact_ids;
    std::vector<float> queries;
    std::vector<float> candidates;
};

template <typename T>
void read_exact(std::ifstream &input, T *data, std::size_t count, const char *label) {
    input.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) {
        throw std::runtime_error(std::string("short read: ") + label);
    }
}

Fixture load_fixture(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open fixture: " + path);
    }

    char magic[4];
    read_exact(input, magic, 4, "magic");
    if (std::memcmp(magic, "SCK1", 4) != 0) {
        throw std::runtime_error("invalid fixture magic");
    }

    Fixture fixture;
    read_exact(input, &fixture.query_count, 1, "query_count");
    read_exact(input, &fixture.candidate_count, 1, "candidate_count");
    read_exact(input, &fixture.dimension, 1, "dimension");
    read_exact(input, &fixture.topk, 1, "topk");
    if (fixture.dimension != 512 || fixture.candidate_count < fixture.topk) {
        throw std::runtime_error("unsupported fixture shape");
    }

    fixture.candidate_ids.resize(
        static_cast<std::size_t>(fixture.query_count) * fixture.candidate_count);
    fixture.exact_ids.resize(static_cast<std::size_t>(fixture.query_count) * fixture.topk);
    fixture.queries.resize(static_cast<std::size_t>(fixture.query_count) * fixture.dimension);
    fixture.candidates.resize(
        static_cast<std::size_t>(fixture.query_count) * fixture.candidate_count * fixture.dimension);
    read_exact(input, fixture.candidate_ids.data(), fixture.candidate_ids.size(), "candidate_ids");
    read_exact(input, fixture.exact_ids.data(), fixture.exact_ids.size(), "exact_ids");
    read_exact(input, fixture.queries.data(), fixture.queries.size(), "queries");
    read_exact(input, fixture.candidates.data(), fixture.candidates.size(), "candidates");
    return fixture;
}

std::size_t overlap_count(
    const std::vector<std::int64_t> &left, const std::vector<std::int64_t> &right) {
    std::unordered_set<std::int64_t> values(left.begin(), left.end());
    std::size_t count = 0;
    for (auto value : right) {
        count += values.count(value);
    }
    return count;
}

std::vector<std::size_t> descending_order(const std::vector<double> &scores) {
    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        return scores[left] > scores[right];
    });
    return order;
}

DiagBlock make_block(
    const SEALContext &context,
    const float *candidate_base,
    std::size_t fixture_candidate_count,
    std::size_t pack_offset,
    std::size_t pack_count,
    std::size_t block_begin,
    std::size_t block_end,
    double scale,
    bool force_full_dimension,
    bool dense_simd,
    bool corrected_dense_diagonal,
    bool full_bsgs,
    std::size_t baby_size) {
    constexpr std::size_t slots = 4096;
    constexpr std::size_t stride = 512;
    CKKSEncoder encoder(context);

    DiagBlock block;
    block.block_id = "sift";
    block.snapshot_id = "sift1m";
    block.layout.slots = slots;
    block.layout.stride = stride;
    block.layout.packing = full_bsgs ? "offset-major-bsgs-v1" : "offset-major";
    block.layout.poly_modulus_degree = 8192;
    block.layout.scale = scale;

    for (std::size_t diagonal = block_begin; diagonal < block_end; ++diagonal) {
        std::vector<double> encoded(slots, 0.0);
        bool has_nonzero_value = false;
        for (std::size_t local = 0; local < pack_count; ++local) {
            const std::size_t candidate = pack_offset + local;
            if (candidate >= fixture_candidate_count) {
                break;
            }
            const std::size_t output_slot = dense_simd ? local : local * stride;
            // With q repeated every 512 slots, rotate(q, diagonal)[s] is
            // q[(s + diagonal) mod 512].  Dense packing must therefore place
            // x_s[(s + diagonal) mod 512] on diagonal `diagonal`.
            const std::size_t component = dense_simd && corrected_dense_diagonal
                ? (output_slot + diagonal) % stride
                : diagonal;
            const std::size_t source = candidate * stride + component;
            double value = static_cast<double>(candidate_base[source]);
            // SIFT has only 129 active augmented dimensions.  For performance
            // measurements, populate the remaining candidate coordinates while
            // leaving the corresponding query coordinates at zero.  This keeps
            // the exact score unchanged but exercises the complete 512-D circuit.
            if (force_full_dimension && component >= 129 && value == 0.0) {
                value = 0.01 * static_cast<double>(1 + ((candidate + diagonal) % 7));
            }
            encoded[output_slot] = value;
            has_nonzero_value = has_nonzero_value || value != 0.0;
        }
        // Sparse SIFT descriptors can make an entire candidate diagonal zero.
        // Skipping it is mathematically exact and avoids SEAL's transparent-
        // ciphertext safety exception on multiply_plain(..., 0).
        if (!has_nonzero_value) continue;
        if (full_bsgs) {
            const std::size_t giant = (diagonal / baby_size) * baby_size;
            if (giant != 0) {
                std::vector<double> adjusted(slots);
                const std::size_t shift = giant % slots;
                // SEAL Rot(v,+g)[s] = v[(s+g) mod slots].  Store Rot(v,-g)
                // so that one outer Rot(+g) aligns the whole giant group.
                for (std::size_t slot = 0; slot < slots; ++slot) {
                    adjusted[slot] = encoded[(slot + slots - shift) % slots];
                }
                encoded.swap(adjusted);
            }
        }
        Plaintext plaintext;
        encoder.encode(encoded, scale, plaintext);
        block.layout.diag_offsets.push_back(static_cast<int>(diagonal));
        block.diag_plaintexts.push_back(std::move(plaintext));
    }
    return block;
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::string fixture_path;
        std::string details_path = "results/sift1m/ckks_details.csv";
        std::string summary_path = "results/sift1m/ckks_summary.json";
        std::size_t requested_queries = 0;
        bool force_full_dimension = false;
        bool dense_simd = false;
        bool corrected_dense_diagonal = false;
        bool full_bsgs = false;
        for (int i = 1; i < argc; ++i) {
            std::string argument = argv[i];
            auto need = [&]() -> std::string {
                if (++i >= argc) {
                    throw std::runtime_error("missing argument value");
                }
                return argv[i];
            };
            if (argument == "--fixture") fixture_path = need();
            else if (argument == "--details") details_path = need();
            else if (argument == "--summary") summary_path = need();
            else if (argument == "--queries") requested_queries = std::stoul(need());
            else if (argument == "--full-dimension") force_full_dimension = true;
            else if (argument == "--dense-simd") {
                dense_simd = true;
                corrected_dense_diagonal = true;
            }
            else if (argument == "--legacy-dense-simd") dense_simd = true;
            else if (argument == "--full-bsgs") {
                dense_simd = true;
                corrected_dense_diagonal = true;
                full_bsgs = true;
            }
            else throw std::runtime_error("unknown argument: " + argument);
        }
        if (fixture_path.empty()) {
            throw std::runtime_error("usage: ckks_sift_rerank --fixture FILE [--queries N]");
        }

        Fixture fixture = load_fixture(fixture_path);
        const std::size_t query_count = requested_queries
            ? std::min<std::size_t>(requested_queries, fixture.query_count)
            : fixture.query_count;
        constexpr std::size_t slots = 4096;
        constexpr std::size_t stride = 512;
        constexpr std::size_t baby_size = 32;
        const std::size_t pack_size = dense_simd ? slots : slots / stride;
        const double scale = std::pow(2.0, 40);

        EncryptionParameters parameters(scheme_type::ckks);
        parameters.set_poly_modulus_degree(8192);
        parameters.set_coeff_modulus(CoeffModulus::Create(8192, {60, 40, 40, 60}));
        SEALContext context(parameters, true);
        KeyGenerator keygen(context);
        SecretKey secret_key = keygen.secret_key();
        PublicKey public_key;
        RelinKeys relin_keys;
        keygen.create_public_key(public_key);
        keygen.create_relin_keys(relin_keys);

        std::vector<int> rotation_steps;
        for (int step = 1; step < 32; ++step) rotation_steps.push_back(step);
        for (int step = 32; step <= 480; step += 32) rotation_steps.push_back(step);
        GaloisKeys galois_keys;
        keygen.create_galois_keys(rotation_steps, galois_keys);

        BSGSPlan plan;
        for (int step = 0; step < 32; ++step) plan.baby.push_back(step);
        for (int step = 0; step <= 480; step += 32) plan.giant.push_back(step);

        HeComputeService service(context);
        service.EvalKeys("sift-client", "v1", galois_keys, relin_keys);
        CKKSEncoder encoder(context);
        Encryptor encryptor(context, public_key);
        Decryptor decryptor(context, secret_key);

        std::ofstream details(details_path);
        if (!details) throw std::runtime_error("cannot open details output");
        details << "query,candidate_rank,candidate_id,plain_score,ckks_score,abs_error\n";
        details << std::setprecision(12);

        double plain_recall_sum = 0.0;
        double ckks_recall_sum = 0.0;
        double error_sum = 0.0;
        double max_error = 0.0;
        double total_eval_ms = 0.0;
        double total_wall_ms = 0.0;
        std::uint64_t total_response_bytes = 0;
        std::uint64_t total_query_ciphertext_bytes = 0;
        std::uint64_t total_rotations = 0;
        std::uint64_t total_multiplications = 0;
        std::size_t exact_order_queries = 0;
        std::size_t exact_set_queries = 0;

        for (std::size_t query_index = 0; query_index < query_count; ++query_index) {
            auto query_started = std::chrono::steady_clock::now();
            const float *query = fixture.queries.data() + query_index * fixture.dimension;
            const float *candidate_base = fixture.candidates.data()
                + query_index * fixture.candidate_count * fixture.dimension;

            std::vector<double> query_slots(slots);
            for (std::size_t slot = 0; slot < slots; ++slot) {
                query_slots[slot] = query[slot % stride];
            }
            Plaintext query_plaintext;
            Ciphertext encrypted_query;
            encoder.encode(query_slots, scale, query_plaintext);
            encryptor.encrypt(query_plaintext, encrypted_query);
            total_query_ciphertext_bytes += encrypted_query.save_size(compr_mode_type::none);

            std::vector<double> ckks_scores(fixture.candidate_count, 0.0);
            std::vector<double> plain_scores(fixture.candidate_count, 0.0);
            for (std::size_t candidate = 0; candidate < fixture.candidate_count; ++candidate) {
                const float *vector = candidate_base + candidate * fixture.dimension;
                double score = 0.0;
                for (std::size_t dimension = 0; dimension < fixture.dimension; ++dimension) {
                    score += static_cast<double>(query[dimension]) * vector[dimension];
                }
                plain_scores[candidate] = score;
            }

            for (std::size_t pack_offset = 0; pack_offset < fixture.candidate_count;
                 pack_offset += pack_size) {
                const std::size_t pack_count = std::min<std::size_t>(
                    pack_size, fixture.candidate_count - pack_offset);
                ScoreBatchRequest request;
                request.client_id = "sift-client";
                request.key_ver = "v1";
                request.ct_q = encrypted_query;
                request.bsgs = plan;
                request.scale = scale;
                const std::size_t active_end = force_full_dimension ? 512 : 129;
                for (std::size_t block_begin = 0; block_begin < active_end; block_begin += 128) {
                    request.blocks.push_back(make_block(
                        context, candidate_base, fixture.candidate_count,
                        pack_offset, pack_count, block_begin,
                        std::min<std::size_t>(block_begin + 128, active_end),
                        scale, force_full_dimension, dense_simd,
                        corrected_dense_diagonal, full_bsgs, baby_size));
                }

                auto reply = service.ScoreBatch(request);
                total_eval_ms += static_cast<double>(reply.telemetry.lat_us) / 1000.0;
                total_rotations += reply.telemetry.rot_cnt;
                total_multiplications += reply.telemetry.mul_cnt;
                for (const auto &ciphertext : reply.scores_ciphertexts) {
                    total_response_bytes += ciphertext.save_size(compr_mode_type::none);
                    Plaintext decrypted;
                    std::vector<double> decoded;
                    decryptor.decrypt(ciphertext, decrypted);
                    encoder.decode(decrypted, decoded);
                    for (std::size_t local = 0; local < pack_count; ++local) {
                        const std::size_t output_slot = dense_simd ? local : local * stride;
                        ckks_scores[pack_offset + local] += decoded[output_slot] / scale;
                    }
                }
            }

            std::vector<std::size_t> plain_order = descending_order(plain_scores);
            std::vector<std::size_t> ckks_order = descending_order(ckks_scores);
            std::vector<std::int64_t> exact_topk;
            std::vector<std::int64_t> plain_topk;
            std::vector<std::int64_t> ckks_topk;
            for (std::size_t rank = 0; rank < fixture.topk; ++rank) {
                exact_topk.push_back(fixture.exact_ids[query_index * fixture.topk + rank]);
                plain_topk.push_back(fixture.candidate_ids[
                    query_index * fixture.candidate_count + plain_order[rank]]);
                ckks_topk.push_back(fixture.candidate_ids[
                    query_index * fixture.candidate_count + ckks_order[rank]]);
            }
            plain_recall_sum += static_cast<double>(overlap_count(plain_topk, exact_topk)) / fixture.topk;
            ckks_recall_sum += static_cast<double>(overlap_count(ckks_topk, exact_topk)) / fixture.topk;
            exact_order_queries += plain_topk == ckks_topk;
            exact_set_queries += overlap_count(plain_topk, ckks_topk) == fixture.topk;

            for (std::size_t candidate = 0; candidate < fixture.candidate_count; ++candidate) {
                const double error = std::abs(plain_scores[candidate] - ckks_scores[candidate]);
                error_sum += error;
                max_error = std::max(max_error, error);
                details << query_index << ',' << candidate + 1 << ','
                        << fixture.candidate_ids[query_index * fixture.candidate_count + candidate] << ','
                        << plain_scores[candidate] << ',' << ckks_scores[candidate] << ',' << error << '\n';
            }
            auto query_finished = std::chrono::steady_clock::now();
            total_wall_ms += std::chrono::duration<double, std::milli>(
                query_finished - query_started).count();
            if ((query_index + 1) % 10 == 0 || query_index + 1 == query_count) {
                std::cout << "processed " << query_index + 1 << '/' << query_count << std::endl;
            }
        }

        const double denominator = static_cast<double>(query_count);
        const double score_count = denominator * fixture.candidate_count;
        std::ofstream summary(summary_path);
        if (!summary) throw std::runtime_error("cannot open summary output");
        summary << std::fixed << std::setprecision(10)
                << "{\n"
                << "  \"query_count\": " << query_count << ",\n"
                << "  \"candidate_count\": " << fixture.candidate_count << ",\n"
                << "  \"topk\": " << fixture.topk << ",\n"
                << "  \"padded_dimension\": 512,\n"
                << "  \"active_dimension\": " << (force_full_dimension ? 512 : 129) << ",\n"
                << "  \"packing\": \""
                << (dense_simd
                    ? (full_bsgs
                        ? "dense-corrected-full-bsgs"
                        : (corrected_dense_diagonal ? "dense-corrected" : "dense-legacy"))
                    : "stride-512") << "\",\n"
                << "  \"mean_plain_recall_at_10\": " << plain_recall_sum / denominator << ",\n"
                << "  \"mean_ckks_recall_at_10\": " << ckks_recall_sum / denominator << ",\n"
                << "  \"top10_order_consistency_rate\": "
                << static_cast<double>(exact_order_queries) / denominator << ",\n"
                << "  \"top10_set_consistency_rate\": "
                << static_cast<double>(exact_set_queries) / denominator << ",\n"
                << "  \"mean_absolute_score_error\": " << error_sum / score_count << ",\n"
                << "  \"max_absolute_score_error\": " << max_error << ",\n"
                << "  \"mean_he_eval_ms_per_query\": " << total_eval_ms / denominator << ",\n"
                << "  \"mean_local_wall_ms_per_query\": " << total_wall_ms / denominator << ",\n"
                << "  \"query_ciphertext_bytes_per_query\": "
                << static_cast<double>(total_query_ciphertext_bytes) / denominator << ",\n"
                << "  \"response_ciphertext_bytes_per_query\": "
                << static_cast<double>(total_response_bytes) / denominator << ",\n"
                << "  \"mean_rotations_per_query\": "
                << static_cast<double>(total_rotations) / denominator << ",\n"
                << "  \"mean_multiplications_per_query\": "
                << static_cast<double>(total_multiplications) / denominator << "\n"
                << "}\n";
        std::cout << "summary written to " << summary_path << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "fatal: " << error.what() << std::endl;
        return 1;
    }
}
