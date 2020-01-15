project(TestInstallation)
cmake_minimum_required(VERSION 3.5)

find_package(MyMath 1.0)

add_executable(mymathApp mymathApp.cc)
target_link_libraries(mymathApp MyMath::mymath)