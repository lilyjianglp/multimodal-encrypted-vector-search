cmake_minimum_required(VERSION 3.16)
project(hecompute_compute CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Protobuf / protoc ----
find_package(Protobuf REQUIRED)
find_program(PROTOC protoc REQUIRED)

# ---- gRPC（优先用 CONFIG，找不到则降级到系统库 + 手找插件）----
find_package(gRPC QUIET)
if (gRPC_FOUND)
  set(GRPCPP_TARGET gRPC::grpc++)
  # 有些发行版没导出 gRPC::grpc_cpp_plugin 这个 target；直接找可执行文件
  find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  message(STATUS "gRPC CONFIG not found -> fallback to system libs")
  find_library(GRPCPP_LIB grpc++)
  find_library(GRPC_LIB grpc)
  find_path(GRPC_INCLUDE_DIR grpcpp/grpcpp.h)
  add_library(grpcpp INTERFACE)
  target_include_directories(grpcpp INTERFACE ${GRPC_INCLUDE_DIR})
  target_link_libraries(grpcpp INTERFACE ${GRPCPP_LIB} ${GRPC_LIB})
  set(GRPCPP_TARGET grpcpp)
  find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
endif()

if (NOT GRPC_CPP_PLUGIN_EXECUTABLE)
  message(FATAL_ERROR "grpc_cpp_plugin not found. Install 'protobuf-compiler-grpc'.")
endif()

# ---- SEAL：优先 find_package，否则手工 SEAL_LIB/SEAL_INCLUDE_DIR ----
find_package(SEAL QUIET)
if (SEAL_FOUND)
  set(SEAL_TARGET SEAL::seal)
else()
  find_library(SEAL_LIB seal PATHS ${SEAL_LIB_DIR})
  find_path(SEAL_INCLUDE_DIR seal/seal.h PATHS ${SEAL_INCLUDE_DIR})
  if (NOT SEAL_LIB OR NOT SEAL_INCLUDE_DIR)
    message(FATAL_ERROR "Microsoft SEAL not found. Set SEAL_LIB and SEAL_INCLUDE_DIR or install SEAL.")
  endif()
  add_library(SEAL_IMPORTED INTERFACE)
  target_include_directories(SEAL_IMPORTED INTERFACE ${SEAL_INCLUDE_DIR})
  target_link_libraries(SEAL_IMPORTED INTERFACE ${SEAL_LIB})
  set(SEAL_TARGET SEAL_IMPORTED)
endif()

# ---- 生成 hecompute gRPC/Proto ----
set(PROTO_DIR ${CMAKE_SOURCE_DIR}/../proto)
set(HE_PROTO  ${PROTO_DIR}/hecompute.proto)
set(GEN_DIR   ${CMAKE_BINARY_DIR}/_gen)
file(MAKE_DIRECTORY ${GEN_DIR})

add_custom_command(
  OUTPUT ${GEN_DIR}/hecompute.pb.cc ${GEN_DIR}/hecompute.pb.h
         ${GEN_DIR}/hecompute.grpc.pb.cc ${GEN_DIR}/hecompute.grpc.pb.h
  COMMAND ${PROTOC}
          -I ${PROTO_DIR}
          --cpp_out=${GEN_DIR}
          --grpc_out=${GEN_DIR}
          --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
          ${HE_PROTO}
  DEPENDS ${HE_PROTO}
  VERBATIM
)

add_library(he_proto
  ${GEN_DIR}/hecompute.pb.cc
  ${GEN_DIR}/hecompute.grpc.pb.cc
)
target_include_directories(he_proto PUBLIC ${GEN_DIR})
target_link_libraries(he_proto PUBLIC ${GRPCPP_TARGET} protobuf::libprotobuf)

# ---- 你的核心实现 ----
add_library(he_core
  hecompute_service.cpp
  hecompute_service.hpp
)
target_include_directories(he_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(he_core PUBLIC ${SEAL_TARGET})

# ---- Server 可执行文件 ----
add_executable(hecompute_server server_main.cpp)
target_link_libraries(hecompute_server PRIVATE he_core he_proto ${GRPCPP_TARGET} protobuf::libprotobuf ${SEAL_TARGET})

# ---- Demo（你原 main.cpp）----
add_executable(hecompute_demo main.cpp)
target_link_libraries(hecompute_demo PRIVATE he_core ${SEAL_TARGET})
