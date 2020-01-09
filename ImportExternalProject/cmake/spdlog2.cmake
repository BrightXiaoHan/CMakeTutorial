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