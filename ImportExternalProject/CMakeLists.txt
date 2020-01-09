project(ImportExternalProject)
cmake_minimum_required(VERSION 3.11)

add_definitions(-std=c++11)  # 指定采用c++11进行编译（spdlog需要c++11）
add_executable(test_spdlog testspdlog.cc)

# 测试时可以取消注释相应的内容

######################################
# 采用添加submodules的方式引入项目     #
######################################

# add_subdirectory(spdlog)
# add_executable(test_spdlog testspdlog.cc)
# target_link_libraries(test_spdlog spdlog::spdlog)

########################################
# 采用ExternalProject_Add的方式引入项目 #
#######################################

# set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
# include(spdlog)

# target_link_libraries(test_spdlog ${SPDLOG_LIB})
# target_include_directories(test_spdlog PRIVATE ${SPDLOG_INCLUDE_DIR})

########################################
# 采用ExternalProject_Add的方式引入项目 #
#######################################
# set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
# include(spdlog2)

# target_link_libraries(test_spdlog PRIVATE spdlog)