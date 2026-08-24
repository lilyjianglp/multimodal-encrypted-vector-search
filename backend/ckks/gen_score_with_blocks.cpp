// gen_score_with_blocks.cpp
#include <seal/seal.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <iostream>
using namespace seal;

static std::string b64(const std::string &bytes){
    static const char tbl[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; out.reserve((bytes.size()*4+2)/3);
    uint32_t val=0; int valb=-6;
    for (unsigned char c: bytes){ val=(val<<8)+c; valb+=8; while(valb>=0){ out.push_back(tbl[(val>>valb)&0x3F]); valb-=6; } }
    if (valb>-6) out.push_back(tbl[((val<<8)>>(valb+8))&0x3F]);
    while(out.size()%4) out.push_back('=');
    return out;
}

static std::string save_pt_b64(const Plaintext &pt){
    std::ostringstream os; pt.save(os); return b64(os.str());
}

int main(){
    // 1) 参数与服务端一致
    size_t N = 8192; std::vector<int> bits={60,40,40,60}; double SCALE = std::pow(2.0,40);
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
    SEALContext ctx(parms, true);

    // 2) 读 ct_q.b64
    std::ifstream fctb64("ct_q.b64"); if(!fctb64.good()){ std::cerr<<"ct_q.b64 not found\n"; return 2; }
    std::string ct_q_b64((std::istreambuf_iterator<char>(fctb64)), std::istreambuf_iterator<char>());
    // 去空白
    ct_q_b64.erase(std::remove_if(ct_q_b64.begin(), ct_q_b64.end(), [](unsigned char c){ return std::isspace(c); }), ct_q_b64.end());

    // 3) 造 4 个对角明文（与 diag_offsets=[0,1,2,3] 对齐）
    CKKSEncoder enc(ctx);
    size_t slots = enc.slot_count(); // 4096
    std::vector<double> v0(slots, 1.0);           // 全 1
    std::vector<double> v1(slots);                // 斜坡
    std::vector<double> v2(slots);                // 正弦
    std::vector<double> v3(slots);                // 伪随机
    for (size_t i=0;i<slots;i++){
        v1[i] = (double)i / (double)slots;                   // [0,1)
        v2[i] = std::sin(2*M_PI * (double)i / 64.0);         // 简单周期
    }
    std::mt19937_64 rng(42); std::uniform_real_distribution<double> U(-0.5,0.5);
    for (size_t i=0;i<slots;i++) v3[i] = U(rng);

    Plaintext p0,p1,p2,p3;
    enc.encode(v0, SCALE, p0);
    enc.encode(v1, SCALE, p1);
    enc.encode(v2, SCALE, p2);
    enc.encode(v3, SCALE, p3);

    std::string b0 = save_pt_b64(p0);
    std::string b1 = save_pt_b64(p1);
    std::string b2 = save_pt_b64(p2);
    std::string b3 = save_pt_b64(p3);

    // 4) 拼 JSON（只用 ASCII，base64 不含引号/换行）
    std::ostringstream js;
    js << "{"
       << "\"client_id\":\"gw\","
       << "\"key_ver\":\"v1\","
       << "\"ct_q\":\"" << ct_q_b64 << "\","
       << "\"scale\":1099511627776,"
       << "\"out_ct_count\":8,"
       << "\"out_ct_bytes\":16384,"
       << "\"pack_slots\":4096,"
       << "\"bsgs\":{\"baby\":[0,1,2,3],\"giant\":[0,8,16,24]},"
       << "\"blocks\":[{"
           << "\"block_id\":\"snap0_blk0\","
           << "\"snapshot_id\":\"snap0\","
           << "\"layout\":{"
               << "\"slots\":4096,"
               << "\"stride\":4096,"
               << "\"diag_offsets\":[0,1,2,3],"
               << "\"packing\":\"offset-major\","
               << "\"poly_modulus_degree\":8192,"
               << "\"scale\":1099511627776,"
               << "\"level\":0"
           << "},"
           << "\"diag_plaintexts\":["
               << "\"" << b0 << "\","
               << "\"" << b1 << "\","
               << "\"" << b2 << "\","
               << "\"" << b3 << "\""
           << "]"
       << "}]"
    << "}";

    // 5) 写到桌面，便于直接 curl
    std::ofstream ofs("/home/wen/Desktop/score_with_blocks.json");
    ofs << js.str(); ofs.close();
    std::cout << "wrote /home/wen/Desktop/score_with_blocks.json\n";
    return 0;
}
