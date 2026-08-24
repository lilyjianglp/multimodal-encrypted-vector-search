#include "hecompute_service.hpp"
#include <seal/seal.h>
#include <iostream>
#include <random>

using namespace seal;
using namespace std;

// 用随机向量生成一个“对角素材块”：offset-major 的明文对角
static DiagBlock MakeMockDiagBlock(const SEALContext& ctx,
                                   size_t slots, size_t dim,
                                   const std::vector<int>& diag_offsets,
                                   double scale)
{
    CKKSEncoder encoder(ctx);
    std::mt19937_64 rng(2025);
    std::normal_distribution<double> dist(0.0, 1.0);

    DiagBlock blk;
    blk.block_id = "blk-0001";
    blk.snapshot_id = "snap-1";
    blk.layout.slots = slots;
    blk.layout.stride = 1;
    blk.layout.diag_offsets = diag_offsets;
    blk.layout.packing = "offset-major";
    blk.layout.poly_modulus_degree = ctx.first_context_data()->parms().poly_modulus_degree();
    blk.layout.scale = scale;
    blk.layout.level = static_cast<int>(ctx.first_context_data()->chain_index());

    // 为简单：每个对角素材就是长度=slots 的明文向量（前 dim 有值，后面补零）
    for (size_t k = 0; k < diag_offsets.size(); ++k) {
        vector<double> w(slots, 0.0);
        for (size_t i = 0; i < dim; ++i) w[i] = dist(rng);
        Plaintext pt; encoder.encode(w, scale, pt);
        blk.diag_plaintexts.push_back(std::move(pt));
    }
    return blk;
}

int main() {
    // 参数
    size_t N = 8192;
    double scale = pow(2.0, 40);
    size_t dim = 4096;
    size_t slots = N / 2;

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, {60, 40, 40, 60}));
    SEALContext context(parms);

    // 生成密钥 & 旋转密钥
    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();
    PublicKey pk; keygen.create_public_key(pk);
    RelinKeys relin; keygen.create_relin_keys(relin);

    // BSGS 计划（示例）：giant 步长 32，baby 覆盖 0..31
    BSGSPlan plan;
    for (int g = -128; g <= 128; g += 32) plan.giant.push_back(g);
    for (int b = 0; b < 32; ++b) plan.baby.push_back(b);

    // 旋转密钥覆盖 giant ∪ baby
    std::vector<int> rot_steps;
    rot_steps.insert(rot_steps.end(), plan.baby.begin(), plan.baby.end());
    rot_steps.insert(rot_steps.end(), plan.giant.begin(), plan.giant.end());
    // 去重
    std::sort(rot_steps.begin(), rot_steps.end());
    rot_steps.erase(std::unique(rot_steps.begin(), rot_steps.end()), rot_steps.end());
    GaloisKeys gal; keygen.create_galois_keys(rot_steps, gal);

    // 客户端：生成 q 并加密
    CKKSEncoder encoder(context);
    Encryptor encryptor(context, pk);
    Decryptor decryptor(context, sk);
    Evaluator evaluator(context);

    std::mt19937_64 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    vector<double> q(slots, 0.0);
    for (size_t i = 0; i < dim; ++i) q[i] = dist(rng);
    Plaintext pt_q; encoder.encode(q, scale, pt_q);
    Ciphertext ct_q; encryptor.encrypt(pt_q, ct_q);

    // 服务：注册密钥
    HeComputeService::SetNumThreads(8);
    HeComputeService svc(context);
    svc.EvalKeys("clientA", "v1", gal, relin);

    // 构造一个 block（diag_offsets 就用 0..31 做示例）
    std::vector<int> diag_offsets;
    for (int t = 0; t < 32; ++t) diag_offsets.push_back(t);
    DiagBlock blk = MakeMockDiagBlock(context, slots, dim, diag_offsets, scale);

    // 组装请求
    ScoreBatchRequest req;
    req.client_id = "clientA";
    req.key_ver = "v1";
    req.ct_q = ct_q;
    req.blocks = { blk };
    req.bsgs = plan;
    req.scale = scale;

    // 计算
    auto reply = svc.ScoreBatch(req);

    // 解密查看第一个 block 的部分槽位（这里只是演示：并未做“对半求和归并”）
    Plaintext pt;
    decryptor.decrypt(reply.scores_ciphertexts[0], pt);
    std::vector<double> res; encoder.decode(pt, res);

    std::cout << "[OK] block=" << blk.block_id
              << " pack_shape=[" << reply.pack_shapes[0][0] << "," << reply.pack_shapes[0][1] << "]\n";
    std::cout << "telemetry: lat_us=" << reply.telemetry.lat_us
              << " rot=" << reply.telemetry.rot_cnt
              << " mul=" << reply.telemetry.mul_cnt << std::endl;

    // 如果你希望每块输出“单一分数槽位”，在 DotDiag_BSGS 末尾按 stride 做对半求和再返回
    return 0;
}
