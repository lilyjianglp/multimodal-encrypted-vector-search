cmake_minimum_required(VERSION 3.13)
project(ckks_dump LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(SEAL REQUIRED)
add_executable(ckks_decrypt_dump ckks_decrypt_dump.cpp)
target_link_libraries(ckks_decrypt_dump SEAL::seal)

add_executable(ckks_keygen ckks_keygen.cpp)
target_link_libraries(ckks_keygen SEAL::seal)

add_executable(ckks_make_ctq_from_npy ckks_make_ctq_from_npy.cpp)
target_link_libraries(ckks_make_ctq_from_npy SEAL::seal)

add_executable(make_dia1_768 make_dia1_768.cpp)
target_link_libraries(make_dia1_768 SEAL::seal)

add_executable(ckks_sift_rerank
  ckks_sift_rerank.cpp
  ../compute/hecompute_service.cpp
)
target_include_directories(ckks_sift_rerank PRIVATE ../compute)
target_link_libraries(ckks_sift_rerank SEAL::seal)
