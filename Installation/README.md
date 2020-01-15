# 安装
本文主要介绍如何将项目生成的库文件、头文件、可执行文件或相关文件等安装到指定位置（系统目录，或发行包目录）。在cmake中，这主要是通过`install`方法在CMakeLists.txt中配置，`make install`命令安装相关文件来实现的。

## 编写一个简单的库
编写一个计算整数和浮点数之和的库函数mymath

mymath.h
```cpp
#ifndef MYMATH_H
#define MYMATH_H

int add(int, int);
double add(double, double);
#endif
```
mymath.cc
```cpp
#include "mymath.h"

int add(int a, int b){
    return a+b;
}

double add(double a, double b){
    return a+b;
}
```
可执行程序mymathApp.cc
```cpp
#include <iostream>
#include "mymath.h"

using namespace std;

int main(int argc, char const *argv[])
{
    double a = add(1.1, 1.1);
    int b = add(1, 1);
    cout << "1.1加1.1等于" << a <<endl;
    cout << "1加1等于" << b <<endl;
    return 0;
}

```
在CMakeLists中添加配置
```cmake
cmake_minimum_required(VERSION 3.0)
project(Installation VERSION 1.0)

# 如果想生成静态库，使用下面的语句
# add_library(mymath mymath.cc)
# target_include_directories(mymath PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 如果想生成动态库，使用下面的语句
add_library(mymath SHARED mymath.cc)
target_include_directories(mymath PRIVATE  ${CMAKE_SOURCE_DIR}/include)
set_target_properties(mymath PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/mymath.h)

# 生成可执行文件
add_executable(mymathapp mymathApp.cc)
target_link_libraries(mymathapp mymath)
target_include_directories(mymathapp PRIVATE ${CMAKE_SOURCE_DIR}/include)

```
接下来我们为生成的target配置安装目录。`install`方法的基础用法如下
```cmake
install(TARGETS MyLib
        EXPORT MyLibTargets 
        LIBRARY DESTINATION lib  # 动态库安装路径
        ARCHIVE DESTINATION lib  # 静态库安装路径
        RUNTIME DESTINATION bin  # 可执行文件安装路径
        PUBLIC_HEADER DESTINATION include  # 头文件安装路径
        )
```
LIBRARY, ARCHIVE, RUNTIME, PUBLIC_HEADER是可选的，可以根据需要进行选择。
DESTINATION后面的路径可以自行制定，根目录默认为`CMAKE_INSTALL_PREFIX`,可以试用`set`方法进行指定，如果使用默认值的话，Unix系统的默认值为 `/usr/local`, Windows的默认值为 `c:/Program Files/${PROJECT_NAME}`。比如字linux系统下若LIBRARY的安装路径指定为`lib`,即为`/usr/local/lib`。所以要安装`mymath mymathapp`我们可以这样写
```cmake
# 将库文件，可执行文件，头文件安装到指定目录
install(TARGETS mymath mymathapp
        EXPORT MyMathTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        PUBLIC_HEADER DESTINATION include
        )
```
他人如果使用我们编写的函数库，安装完成后，希望可以通过```find_package```方法进行引用，这时我们需要怎么做呢。

首先我们需要生成一个`MyMathConfigVersion.cmake`的文件来声明版本信息
```cmake
# 写入库的版本信息
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        MyMathConfigVersion.cmake
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY AnyNewerVersion  # 表示该函数库向下兼容
        )
```
其中`PACKAGE_VERSION`便是我们在`CMakeLists.txt`开头`project(Installation VERSION 1.0)`中声明的版本号

第二步我们将前面`EXPORT MyMathTargets`的信息写入到`MyLibTargets.cmake`文件中, 该文件存放目录为`${CMAKE_INSTALL_PREFIX}/lib/cmake/MyMath`
```cmake
install(EXPORT MyMathTargets
        FILE MyLibTargets.cmake
        NAMESPACE MyMath::
        DESTINATION lib/cmake/MyLib
        )
```
最后我们在源代码目录新建一个`MyMathConfig.cmake.in`文件,用于获取配置过程中的变量，并寻找项目依赖包。如果不一来外部项目的话，可以直接include `MyMathTargets.cmake`文件
```
include(CMakeFindDependencyMacro)

# 如果想要获取Config阶段的变量，可以使用这个
# set(my-config-var @my-config-var@)

# 如果你的项目需要依赖其他的库，可以使用下面语句，用法与find_package相同
find_dependency(MYDEP REQUIRED)

# Any extra setup

# Add the targets file
include("${CMAKE_CURRENT_LIST_DIR}/MyMathTargets.cmake")
```
最后在CMakeLists.txt文件中，配置生成`MyMathTargets.cmake`文件，并一同安装到`${CMAKE_INSTALL_PREFIX}/lib/cmake/MyMath`目录中。
```cmake
configure_file(MyMathConfig.cmake.in MyMathConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/MyMathConfig.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/MyMathConfigVersion.cmake"
        DESTINATION lib/cmake/MyMath
        )
```
最后我们在其他项目中，就可以使用
```cmake
find_package(MyMath VERSION 1.0)
target_linked_library(otherapp MyMath::mymath)
```
来引用我们的函数库了。
