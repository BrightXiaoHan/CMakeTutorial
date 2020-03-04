cmake_minimum_required (VERSION 3.0)
project (PybindMatrix)

find_package (Eigen3 REQUIRED NO_MODULE)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
add_library(matrix python_matrix.cc)
target_compile_features(matrix PRIVATE cxx_std_11)
target_link_libraries (matrix ${PYTHON_LIBRARIES} Eigen3::Eigen)

add_executable(test_eigen test_eigen.cc)
target_link_libraries (test_eigen Eigen3::Eigen)