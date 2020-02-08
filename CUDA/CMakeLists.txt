cmake_minimum_required(VERSION 3.8)
project(CUDA_MAT_MUL LANGUAGES CXX CUDA)

add_library(cudaMatMul cudaMatMul.cu cudaMatMul.h)
target_compile_features(cudaMatMul PUBLIC cxx_std_11)

add_executable(main main.cc)
target_link_libraries(main cudaMatMul)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
find_package(cuBLAS)
add_executable(cublasMatMul cublasMatMul.cu)
target_compile_features(cublasMatMul PRIVATE cxx_std_11)
target_link_libraries(cublasMatMul PRIVATE ${CUBLAS_LIBRARIES})