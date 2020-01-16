# 使用包管理工具管理你的依赖库
当依赖的项目较多时，手动安装相关依赖包较为复杂，并且多个项目多个版本的依赖包安装在系统中及容易造成冲突。若通过submodule的方式引入，下载编译耗时较长，同时也不好管理，对于还在开发过程中的项目来说影像效率。本文主要介绍两个帮助我们管理依赖库的工具的基本使用方法，抛砖引玉。它们分别是[vcpkg](https://github.com/microsoft/vcpkg)，Anaconda(https://www.anaconda.com/)

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
### 使用vcpkg安装opencv并测试
众所周知，编译和安装opencv是一件苦差事，使用vcpkg便会变的十分简单。