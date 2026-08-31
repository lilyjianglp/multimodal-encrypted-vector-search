cmake_minimum_required(VERSION 3.14)
project(secure_pir_gateway CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

# 尝试 gRPC 的 CMake config；没有就走 pkg-config
find_package(gRPC CONFIG QUIET)

# Always regenerate the checked-in protocol with the active Protobuf version.
# The old generated sources were produced by protoc 3.12 and cannot be built
# against current Protobuf headers.
find_program(PROTOC protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin REQUIRED)
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/gateway.proto)
set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})
add_custom_command(
  OUTPUT
    ${GEN_DIR}/gateway.pb.cc
    ${GEN_DIR}/gateway.pb.h
    ${GEN_DIR}/gateway.grpc.pb.cc
    ${GEN_DIR}/gateway.grpc.pb.h
  COMMAND ${PROTOC}
    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
    --cpp_out=${GEN_DIR}
    --grpc_out=${GEN_DIR}
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_EXECUTABLE}
    ${PROTO_FILE}
  DEPENDS ${PROTO_FILE}
  VERBATIM
)

add_executable(gateway
  src/main.cpp
  ${GEN_DIR}/gateway.pb.cc
  ${GEN_DIR}/gateway.grpc.pb.cc
)

# 头文件搜索路径
target_include_directories(gateway PRIVATE
  ${Protobuf_INCLUDE_DIRS}
  ${CURL_INCLUDE_DIRS}
  ${GEN_DIR}
  ${CMAKE_SOURCE_DIR}
)

if (gRPC_FOUND)
  message(STATUS "Using gRPC CONFIG mode")
  target_link_libraries(gateway
    PRIVATE
      gRPC::grpc++
      ${Protobuf_LIBRARIES}
      ${CURL_LIBRARIES}
      Threads::Threads
      dl
      crypto      # ★★★ 关键补充
      ssl         # ★ 可选
  )
else()
  message(WARNING "gRPC CONFIG not found, using pkg-config fallback")
  pkg_check_modules(GRPCPP REQUIRED grpc++)
  target_include_directories(gateway PRIVATE ${GRPCPP_INCLUDE_DIRS})
  target_link_directories(gateway PRIVATE ${GRPCPP_LIBRARY_DIRS})
  target_link_libraries(gateway
    PRIVATE
      ${GRPCPP_LIBRARIES}
      ${Protobuf_LIBRARIES}
      ${CURL_LIBRARIES}
      Threads::Threads
      dl
      crypto      # ★★★ 关键补充
      ssl         # ★ 可选
  )
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(gateway PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
