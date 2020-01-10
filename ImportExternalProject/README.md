# 引入外部项目
本节主要介绍如何引入外部项目源码作为自己项目的Library。与[FindPackage](FindPackage/README.md)引入头文件编译好的库不同的是，本节介绍的方法直接将第三方库源码引入到项目中，编译自己的项目时也会连同第三方库的源码一同编译。特别是当我们使用git等工具引入代码时，我们可以很方便地控制第三方代码的版本，防止本地安装的库文件版本与项目存在冲突。下面我们逐一介绍引入外部项目的几种方式，并对比他们的优劣。(以[spdlog](https://github.com/gabime/spdlog)为例)

## 通过Submodle的方式引入
克隆spdlog作为项目的子项目
```bash
git submodule add https://github.com/gabime/spdlog.git
```
本项目已经添加了submodule，所以在项目根目录执行以下命令初始化
```bash
git submodule init
git submodule update
```
切换到我们需要的版本
```bash
git checkout v1.4.2
```
我们已经clone好了，现在只需要将spdlog作为subdirectory加入CMakeLists.txt当中就行了
```cmake
project(ImportExternalProject)
cmake_minimum_required(VERSION 3.5)

add_definitions(-std=c++11)  # 指定采用c++11进行编译（spdlog需要c++11）
add_subdirectory(spdlog)
```

## 在编译时下载项目并引入
首先新建cmake目录，在目录下创建spdlog.cmake并加入以下内容
```cmake
include(ExternalProject)

set(SPDLOG_ROOT ${CMAKE_BINARY_DIR}/thirdparty/SPDLOG)
set(SPDLOG_GIT_TAG  v1.4.1)  # 指定版本
set(SPDLOG_GIT_URL      https://github.com/gabime/spdlog.git)  # 指定git仓库地址
set(SPDLOG_CONFIGURE    cd ${SPDLOG_ROOT}/src/SPDLOG && cmake -D CMAKE_INSTALL_PREFIX=${SPDLOG_ROOT} .)  # 指定配置指令（注意此处修改了安装目录，否则默认情况下回安装到系统目录）
set(SPDLOG_MAKE         cd ${SPDLOG_ROOT}/src/SPDLOG && make)  # 指定编译指令（需要覆盖默认指令，进入我们指定的SPDLOG_ROOT目录下）
set(SPDLOG_INSTALL      cd ${SPDLOG_ROOT}/src/SPDLOG && make install)  # 指定安装指令（需要覆盖默认指令，进入我们指定的SPDLOG_ROOT目录下）

ExternalProject_Add(SPDLOG
        PREFIX            ${SPDLOG_ROOT}
        GIT_REPOSITORY    ${SPDLOG_GIT_URL}
        GIT_TAG           ${SPDLOG_GIT_TAG}
        CONFIGURE_COMMAND ${SPDLOG_CONFIGURE}
        BUILD_COMMAND     ${SPDLOG_MAKE}
        INSTALL_COMMAND   ${SPDLOG_INSTALL}
)

# 指定编译好的静态库文件的路径
set(SPDLOG_LIB       ${SPDLOG_ROOT}/lib/spdlog/libspdlog.a)
# 指定头文件所在的目录
set(SPDLOG_INCLUDE_DIR   ${SPDLOG_ROOT}/include)
```
需要注意的是，我们此处指定了配置、编译、安装的指令，如果不这样做的话，cmake默认将编译好的库和头文件安装在系统目录，而我们希望他安装在build目录下的指定位置。最后我们设置了静态库文件的位置和头文件所在目录，这样我们在CMakeLists.txt中就可以通过target_link_libraries和target_include_directories进行引用了。
```cmake
project(ImportExternalProject)
cmake_minimum_required(VERSION 3.5)

add_definitions(-std=c++11)  # 指定采用c++11进行编译（spdlog需要c++11）
add_executable(test_spdlog testspdlog.cc)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(spdlog)

target_link_libraries(test_spdlog ${SPDLOG_LIB})
target_include_directories(test_spdlog PRIVATE ${SPDLOG_INCLUDE_DIR})
```

## 使用FetchContent (CMake 3.11+)
细心的朋友已经发现了，上述使用ExternalProject_Add的方式引入spdlog是在编译时进行的，也就是说在Configure过程中我们并没有从git上下载项目，这就导致我们无法像submodule那样直接通过add_subdirectory引入项目，而是要预先定义好编译后库文件和头文件的位置，通过target_link_libraries和target_include_directories进行引入。在cmake3.11以及以上的版本，cmake又为我们提供了一种Configure过程引入外部项目的方法--FetchContent。

使用FetchContent的步骤总结起来就是:

- 使用FetchContent_Declare(MyName) 获取项目。可以是一个URL也可以是一个Git仓库。
- 使用FetchContent_GetProperties(MyName) 获取我们需要的变量MyName_*。
- 使用add_subdirectory(${MyName_SOURCE_DIR} ${MyName_BINARY_DIR})引入项目。

在cmake3.14版本，官方又为我们提供了更方便的FetchContent_MakeAvailable方法，将步骤2，3集成在了一起。为了兼容3.11版本，我们可以把它封装成一个宏，这样我们就可以统一使用FetchContent_MakeAvailable方法了。
```cmake
# Campatible with cmake 3.11 and above.
macro(FetchContent_MakeAvailable NAME)
FetchContent_GetProperties(${NAME})
if(NOT ${NAME}_POPULATED)
    FetchContent_Populate(${NAME})
    add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR})
endif()
endmacro()macro(FetchContent_MakeAvailable NAME)
    FetchContent_GetProperties(${NAME})
    if(NOT ${NAME}_POPULATED)
        FetchContent_Populate(${NAME})
        add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR})
    endif()
endmacro()
```
与前面类似，我们在cmake目录下新建spdlog2.cmake，使用FetchContent引入spdlog
```cmake
# 添加第三方依赖包
include(FetchContent)
# FetchContent_MakeAvailable was not added until CMake 3.14
if(${CMAKE_VERSION} VERSION_LESS 3.14)
    include(add_FetchContent_MakeAvailable.cmake)
endif()

set(SPDLOG_GIT_TAG  v1.4.1)  # 指定版本
set(SPDLOG_GIT_URL  https://github.com/gabime/spdlog.git)  # 指定git仓库地址

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY    ${SPDLOG_GIT_URL}
  GIT_TAG           ${SPDLOG_GIT_TAG}
)

FetchContent_MakeAvailable(spdlog)
```
在CMakeLists.txt中，包含cmake/spdlog2.cmake，便可将spdlog作为library来使用了(主义cmake最小版本应当设置为3.11)
```cmake
project(ImportExternalProject)
cmake_minimum_required(VERSION 3.11)

add_definitions(-std=c++11)  # 指定采用c++11进行编译（spdlog需要c++11）
add_executable(test_spdlog testspdlog.cc)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include(spdlog2)

target_link_libraries(test_spdlog PRIVATE spdlog)
```