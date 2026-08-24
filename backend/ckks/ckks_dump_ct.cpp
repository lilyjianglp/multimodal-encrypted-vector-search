// ckks_dump_ct.cpp — 直接解密并 decode 一个 Ciphertext（例如 ct_q.bin）
// 用法：
//   g++ -O2 ckks_dump_ct.cpp -o ckks_dump_ct -lseal
//   ./ckks_dump_ct --context ./context.seal --sk ./sk.bin --ct ./ct_q.bin --show 32
#include <seal/seal.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdlib>

using namespace std;
using namespace seal;
namespace fs = std::filesystem;

static vector<uint8_t> read_all(const fs::path& p){
    ifstream f(p, ios::binary);
    if (!f) throw runtime_error("open failed: " + p.string());
    f.seekg(0, ios::end);
    auto n = f.tellg(); f.seekg(0);
    vector<uint8_t> b((size_t)n);
    if (n>0) f.read((char*)b.data(), (size_t)n);
    return b;
}

static void usage(const char* prog){
    cerr <<
      "Usage:\n  " << prog << " --context context.seal --sk sk.bin --ct ct_q.bin [--show K]\n";
}

int main(int argc,char**argv){
    fs::path ctx_p, sk_p, ct_p; int show = 16;
    auto need=[&](int&i){ if(i+1>=argc){ usage(argv[0]); exit(1);} return argv[++i]; };
    for(int i=1;i<argc;i++){
        string a=argv[i];
        if(a=="--context") ctx_p=need(i);
        else if(a=="--sk") sk_p=need(i);
        else if(a=="--ct") ct_p=need(i);
        else if(a=="--show") show=atoi(need(i));
        else { usage(argv[0]); return 1; }
    }
    if(ctx_p.empty()||sk_p.empty()||ct_p.empty()){ usage(argv[0]); return 1; }

    // load context
    EncryptionParameters parms;
    {
        auto buf=read_all(ctx_p);
        string s((const char*)buf.data(), buf.size());
        stringstream ss(s); parms.load(ss);
    }
    SEALContext ctx(parms, /*expand_mod_chain=*/true);

    // load sk
    SecretKey sk;
    {
        auto buf=read_all(sk_p);
        string s((const char*)buf.data(), buf.size());
        stringstream ss(s); sk.load(ctx, ss);
    }

    // load ct
    Ciphertext ct;
    {
        auto buf=read_all(ct_p);
        string s((const char*)buf.data(), buf.size());
        stringstream ss(s); ct.load(ctx, ss);
    }

    auto cd = ctx.get_context_data(ct.parms_id());
    cout << "polyN=" << parms.poly_modulus_degree()
         << " scale=" << ct.scale()
         << " level=" << (cd? (int)cd->chain_index() : -1) << "\n";

    Decryptor dec(ctx, sk);
    CKKSEncoder enc(ctx);
    Plaintext pt; dec.decrypt(ct, pt);
    vector<double> v; enc.decode(pt, v);

    cout.setf(std::ios::fixed); cout.precision(10);
    cout << "slots=" << v.size() << " first" << show << "=[";
    for (int i=0;i<show && i<(int)v.size(); ++i) {
        if (i) cout << ", ";
        cout << v[i];
    }
    cout << "]\n";
    return 0;
}
