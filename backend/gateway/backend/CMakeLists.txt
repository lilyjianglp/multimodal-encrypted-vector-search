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

add_executable(gateway
  src/main.cpp
  src/gateway.pb.cc
  src/gateway.grpc.pb.cc
)

# 头文件搜索路径
target_include_directories(gateway PRIVATE
  ${Protobuf_INCLUDE_DIRS}
  ${CURL_INCLUDE_DIRS}
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
  link_directories(${GRPCPP_LIBRARY_DIRS})
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

