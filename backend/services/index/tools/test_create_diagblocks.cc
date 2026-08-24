// tools/test_create_diagblocks.cc
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

#include <grpcpp/grpcpp.h>
#include "index_service.grpc.pb.h"


static std::vector<uint64_t> parse_ids(const std::string& s) {
    std::vector<uint64_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoull(tok));
    }
    return out;
}

int main(int argc, char** argv) {
    std::string addr = "localhost:50051";
    std::string ids_csv = "1,2,3,4,5,6,7,8,9,10";
    uint32_t pack_slots = 32;

    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a.rfind("--addr=",0)==0) addr = a.substr(7);
        else if (a.rfind("--cand=",0)==0) ids_csv = a.substr(8);
        else if (a.rfind("--slots=",0)==0) pack_slots = (uint32_t)std::stoul(a.substr(8));
    }

    auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
    auto stub = indexsvc::IndexService::NewStub(channel);

    indexsvc::CreateDiagBlocksRequest req;
    for (auto id : parse_ids(ids_csv)) req.add_candidate_ids(id);
    req.set_pack_slots(pack_slots);

    indexsvc::CreateDiagBlocksResponse resp;
    grpc::ClientContext ctx;
    auto status = stub->CreateDiagBlocks(&ctx, req, &resp);
    if (!status.ok()) {
        std::cerr << "RPC failed: " << status.error_message() << " (code " << (int)status.error_code() << ")\n";
        return 2;
    }

    std::cout << "blocks=" << resp.blocks_size() << "\n";
    for (int i=0;i<resp.blocks_size();++i) {
        const auto& b = resp.blocks(i);
        std::cout << "  ["<<i<<"] id="<< b.block_id()
                  << " slots="<< b.slots()
                  << " stride="<< b.stride()
                  << " path="<< b.mmap_path()
                  << " offsets="<< b.diag_offsets_size()
                  << "\n";
        if (i==0) {
            std::cout << "    first 8 offsets: ";
            for (int j=0;j<std::min(8, b.diag_offsets_size()); ++j) {
                std::cout << b.diag_offsets(j) << (j+1<8? ",":"");
            }
            std::cout << "\n";
        }
    }
    return 0;
}
