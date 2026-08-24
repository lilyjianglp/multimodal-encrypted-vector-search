#include <grpcpp/grpcpp.h>
#include <iostream>
#include <string>
#include "index_service.hpp"

int main(int argc, char** argv) {
    std::string addr = "0.0.0.0:50051";   // 默认
    std::string data_dir = "data";        // 默认

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // 如果第一个参数不是 --xxx，兼容旧写法 ./index_server <data_dir>
        if (i == 1 && arg.rfind("--", 0) != 0) {
            data_dir = arg;
            continue;
        }

        const std::string k_data = "--data_dir=";
        const std::string k_port = "--port=";

        if (arg.rfind(k_data, 0) == 0) {
            data_dir = arg.substr(k_data.size());
        } else if (arg.rfind(k_port, 0) == 0) {
            std::string port = arg.substr(k_port.size());
            addr = "0.0.0.0:" + port;
        }
    }

    std::cout << "[IndexServer] start with data_dir=" << data_dir
              << " addr=" << addr << std::endl;

    IndexServiceImpl service(data_dir);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    std::cout << "IndexService listening on " << addr << std::endl;
    server->Wait();
    return 0;
}

