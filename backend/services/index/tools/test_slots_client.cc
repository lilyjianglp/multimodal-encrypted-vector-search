// tools/test_slots_client.cc
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

#include <grpcpp/grpcpp.h>
#include "index_service.grpc.pb.h"   // 注意：包含的是生成后的短路径

static std::vector<uint64_t> parse_ids(const std::string& csv) {
    std::vector<uint64_t> out;
    std::stringstream ss(csv);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoull(tok));
    }
    return out;
}

int main(int argc, char** argv) {
    std::string addr = "localhost:50051";
    std::string cand_csv = "1,2,3,4,5,6,7,8,9,10";
    uint32_t slots_req = 33;

    for (int i=1; i<argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--addr=",0)==0) addr = a.substr(7);
        else if (a.rfind("--cand=",0)==0) cand_csv = a.substr(8);
        else if (a.rfind("--slots=",0)==0) slots_req = (uint32_t)std::stoul(a.substr(8));
    }

    auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
    auto stub = indexsvc::IndexService::NewStub(channel);

    indexsvc::CreateDiagBlocksRequest req;
    for (auto id : parse_ids(cand_csv)) req.add_candidate_ids(id);
    req.set_pack_slots(slots_req);

    indexsvc::CreateDiagBlocksResponse resp;
    grpc::ClientContext ctx;
    auto status = stub->CreateDiagBlocks(&ctx, req, &resp);
    if (!status.ok()) {
        std::cerr << "RPC failed: " << status.error_message()
                  << " (code " << (int)status.error_code() << ")\n";
        return 2;
    }

    std::cout << "req_slots=" << slots_req
              << "  resp_blocks=" << resp.blocks_size() << "\n";

    for (int i=0;i<resp.blocks_size();++i) {
        const auto& b = resp.blocks(i);
        std::cout << "  ["<<i<<"] block_id="<< b.block_id()
                  << "  slots(returned)="<< b.slots()      // 这就是服务端最终采用的对齐后 slots
                  << "  stride="<< b.stride()
                  << "  offsets="<< b.diag_offsets_size()
                  << "  path="<< b.mmap_path()
                  << "\n";
    }
    return 0;
}
