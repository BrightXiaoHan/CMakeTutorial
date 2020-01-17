# 使用包管理工具管理你的依赖库
当依赖的项目较多时，手动安装相关依赖包较为复杂，并且多个项目多个版本的依赖包安装在系统中及容易造成冲突。若通过submodule的方式引入，下载编译耗时较长，同时也不好管理，对于还在开发过程中的项目来说影像效率。本文主要介绍两个帮助我们管理依赖库的工具的基本使用方法，抛砖引玉。它们分别是[vcpkg](https://github.com/microsoft/vcpkg)，Anaconda(https://www.anaconda.com/)

## 一个简单的小程序
我们写一个简单的展示图片的小程序
```cpp
#include <stdio.h>
#include <opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}
```
在CMakeLists.txt中，我们使用find_package(详见[find_package的使用指南](FindPackage/README.md))来引入opencv的库。
```cmake
cmake_minimum_required(VERSION 3.0)
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
```
接下来我们使用vcpkg与anaconda两个工具安装opencv，并编译我们的项目。

## 使用vcpkg
[vcpkg](https://github.com/microsoft/vcpkg) 是微软开源的一个库管理工具，并原生支持与cmake集成。vcpkg支持常见依赖库的一键安装，并支持cmake通过find_package一键引入。以下只介绍它的基础用法，具体文档请参考github [vcpkg](https://github.com/microsoft/vcpkg)
### 安装vcpkg
```
> git clone https://github.com/Microsoft/vcpkg.git
> cd vcpkg

PS> .\bootstrap-vcpkg.bat
Linux:~/$ ./bootstrap-vcpkg.sh
```

然后，[集成](docs/users/integration.md)至本机环境中，执行 (注意: 首次启动需要管理员权限)
```
PS> .\vcpkg integrate install
Linux:~/$ ./vcpkg integrate install
```

使用以下命令安装任意包
```
PS> .\vcpkg install sdl2 curl
Linux:~/$ ./vcpkg install sdl2 curl
```

## Tab补全/自动补全
`vcpkg`支持在 Powershell 和 bash 中自动补全命令、程序包名称、选项等。如需启用自动补全功能，请使用以下命令:
```
PS> .\vcpkg integrate powershell
Linux:~/$ ./vcpkg integrate bash
```
并重启您的控制台。
最后将vcpkg加入环境变量，便于在任何地方执行vcpkg的命令
### 使用vcpkg安装opencv并测试
众所周知，编译和安装opencv是一件苦差事，使用vcpkg便会变的十分简单。
首先我们查看vcpkg是否支持安装opencv
```bash
vcpkg search opencv
```
确认支持后进行安装
```bash
vcpkg install opencv
```
安装完成后可以使用下面的命令查看安装情况
```
vcpkg list
```
在配置的过程中，将vcpkg.cmake的路径赋值给`CMAKE_TOOLCHAIN_FILE`变量即可。
```bash
VCPKG_HOME=/path/to/your_vcpkg/
cmake .. "-DCMAKE_TOOLCHAIN_FILE=${VCPKG_HOME}/scripts/buildsystems/vcpkg.cmake"
```
如果我们想卸载掉opencv，也只需执行一个命令
```bash
vcpkg remove opencv
```
### 处理安装包的版本问题
如果要指定依赖包的版本，这里不再赘述，请参考微软的博客[Vcpkg: Using multiple enlistments to handle multiple versions of a library](https://devblogs.microsoft.com/cppblog/vcpkg-using-multiple-enlistments/)

## 使用Conda环境
anaconda被大多数人所认为的是一个python的科学计算环境，其实它也可以被用作c,c++虚拟环境的管理。他的优点是可以不使用管理员权限进行安装，并且可以创建多个虚拟环境，不同环境中安装的库之间相互隔离互不干扰。对于需要开发c++扩展的python项目特别适合使用conda环境。
### 安装
国内可以到[清华镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)下载安装包进行安装，如果不需要python环境，可以选择对应的miniconda安装包进行安装。具体安装过程这里不再赘述。
### 创建虚拟环境
与上面一样，假设此时我们需要开发一个名为 DisplayImage 的项目，我们可以为该项目单独建立一个虚拟环境
```bash
conda create -n DisplayImage
```
激活虚拟环境
```bash
source activate DisplayImage
# windows下 
# activate DisplayImage
```
我们可以在当前环境下安装cmake并指定版本3.14，与系统安装的cmake相隔离
```bash
conda install cmake=3.14.0
# 卸载使用conda uninstall cmake
```
接下来我们安装项目的以来库opencv
```bash
conda install opencv
# 卸载使用 conda uninstall opencv
```
配置项目时设置`CMAKE_PREFIX_PATH`变量为虚拟环境的根目录即可
```
cmake .. "-DCMAKE_PREFIX_PATH=/path/to/anaconda/envs/DisplayImage" 
```
