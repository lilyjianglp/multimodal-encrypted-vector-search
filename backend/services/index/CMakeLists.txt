cmake_minimum_required(VERSION 3.16)
project(index_service CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# ---- 找 protoc / grpc_cpp_plugin ----
find_program(PROTOC protoc)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
if(NOT PROTOC OR NOT GRPC_CPP_PLUGIN)
  message(FATAL_ERROR "protoc 或 grpc_cpp_plugin 未找到，先安装 protobuf-compiler 和 protobuf-compiler-grpc")
endif()

# ---- 用 pkg-config 找 gRPC++ ----
pkg_check_modules(GRPCPP REQUIRED grpc++)

# ---- 生成 gRPC / Protobuf 源码 ----
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/index_service.proto)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})

set(PB_CC   ${GEN_DIR}/index_service.pb.cc)
set(PB_H    ${GEN_DIR}/index_service.pb.h)
set(GRPC_CC ${GEN_DIR}/index_service.grpc.pb.cc)
set(GRPC_H  ${GEN_DIR}/index_service.grpc.pb.h)

add_custom_command(
  OUTPUT ${PB_CC} ${PB_H}
  COMMAND ${PROTOC}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
    --cpp_out=${GEN_DIR}
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  VERBATIM)

add_custom_command(
  OUTPUT ${GRPC_CC} ${GRPC_H}
  COMMAND ${PROTOC}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
    --grpc_out=${GEN_DIR}
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE} ${PB_CC} ${PB_H}
  VERBATIM)

add_library(index_proto ${PB_CC} ${GRPC_CC})
target_include_directories(index_proto PUBLIC ${GEN_DIR} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(index_proto PUBLIC ${Protobuf_LIBRARIES} ${GRPCPP_LIBRARIES})
target_include_directories(index_proto PUBLIC ${GRPCPP_INCLUDE_DIRS})

# ---- 业务源码 ----
include_directories(${GEN_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(INDEX_CORE_SRCS
  src/index_service.cc
  src/meta_store.cc
  src/vector_store.cc
  src/diag_export.cc
)

# ---- 服务端可执行 ----
add_executable(index_server
  src/main.cc
  ${INDEX_CORE_SRCS}
)
target_link_libraries(index_server
  PRIVATE index_proto ${GRPCPP_LIBRARIES} ${Protobuf_LIBRARIES}
          sqlite3 faiss
          ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
          OpenMP::OpenMP_CXX)
target_include_directories(index_server PRIVATE ${GRPCPP_INCLUDE_DIRS})

# ---- 离线训练工具 ----
add_executable(train_offline
  src/train_offline.cc
  src/meta_store.cc
)
target_link_libraries(train_offline PRIVATE sqlite3 faiss
  ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} OpenMP::OpenMP_CXX
  ssl crypto)


# ---- 运行目录准备 data 结构（可选）----
add_custom_command(TARGET index_server POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:index_server>/data/diag_blocks
)

# ---- 测试客户端（调用 CreateDiagBlocks）----
add_executable(test_create_diagblocks
  tools/test_create_diagblocks.cc
)
target_include_directories(test_create_diagblocks PRIVATE ${GRPCPP_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(test_create_diagblocks PRIVATE index_proto ${GRPCPP_LIBRARIES} ${Protobuf_LIBRARIES})

# ---- 测试客户端：验证 pack_slots 对齐 ----
add_executable(test_slots_client
  tools/test_slots_client.cc
)
target_include_directories(test_slots_client PRIVATE
  ${GRPCPP_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(test_slots_client PRIVATE
  index_proto ${GRPCPP_LIBRARIES} ${Protobuf_LIBRARIES})


