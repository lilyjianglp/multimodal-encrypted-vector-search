#pragma once
#include "index_service.grpc.pb.h"

class IndexServiceImpl final : public indexsvc::IndexService::Service {
public:
    explicit IndexServiceImpl(std::string data_dir);

    ::grpc::Status GetCenters(::grpc::ServerContext*,
                              const indexsvc::GetCentersRequest*,
                              indexsvc::GetCentersResponse*) override;

    ::grpc::Status GetClusterCandidates(::grpc::ServerContext*,
                              const indexsvc::GetClusterCandidatesRequest*,
                              indexsvc::GetClusterCandidatesResponse*) override;

    ::grpc::Status CreateDiagBlocks(::grpc::ServerContext*,
                              const indexsvc::CreateDiagBlocksRequest*,
                              indexsvc::CreateDiagBlocksResponse*) override;
private:
    std::string data_dir_;
};
