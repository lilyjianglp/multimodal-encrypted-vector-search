#include <seal/seal.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>

using namespace seal;

int main() {
    // 和服务端一致：N=8192, coeff_modulus=[60,40,40,60], scale=2^40
    size_t N = 8192;
    std::vector<int> mod_bits = {60,40,40,60};
    double scale = std::pow(2.0, 40);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, mod_bits));
    SEALContext ctx(parms, /*expand_mod_chain=*/true);

    CKKSEncoder enc(ctx);
    size_t slots = enc.slot_count(); // 4096

    std::vector<double> vals(slots, 1.0); // 全 1
    Plaintext pt;
    enc.encode(vals, scale, pt);

    std::ostringstream os;
    pt.save(os);                          // 注意：保存 Plaintext（不是 Ciphertext）
    std::string bytes = os.str();

    std::ofstream f("pt_one.bin", std::ios::binary);
    f.write(bytes.data(), (std::streamsize)bytes.size());
    f.close();

    std::cout << "[ckks_make_demo_pt] wrote pt_one.bin bytes=" << bytes.size()
              << " slots=" << slots << " scale=2^40\n";
    return 0;
}
