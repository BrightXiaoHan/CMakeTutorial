cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
enable_language(CUDA)
project(cmake_and_cuda)

include(CTest)

add_library(matrix_mul SHARED kernel.cu dev_array.cu)
target_include_directories(matrix_mul PUBLIC ${CMAKE_SOURCE_DIR})

# cuda c++ 支持c++11 标准
# 设置为 PUBLIC属性表示所有链接到这个包的target都采用c++11标准
target_compile_features(matrix_mul PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties( matrix_mul
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )


if(BUILD_TESTING)

  add_executable(matrix_mul_test matrix_mul.cu)
  target_link_libraries(matrix_mul_test PRIVATE matrix_mul)
  set_target_properties(matrix_mul_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  add_test(NAME matrix_mul_test_app COMMAND matrix_mul_test)

endif()
