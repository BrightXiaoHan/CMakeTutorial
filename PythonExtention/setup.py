import os
import shlex
import subprocess
import datetime
import time
import shutil
from setuptools import setup, Extension

cwd = os.path.dirname(os.path.abspath(__file__))


def execute_command(cmdstring, cwd=None, timeout=None, shell=False):
    """执行一个SHELL命令封装了subprocess的Popen方法, 支持超时判断，支持读取stdout和stderr

    Arguments:
        cmdstring {[type]} -- 命令字符创

    Keyword Arguments:
        cwd {str} -- 运行命令时更改路径，如果被设定，子进程会直接先更改当前路径到cwd (default: {None})
        timeout {str} -- 超时时间，秒，支持小数，精度0.1秒 (default: {None})
        shell {bool} -- 是否通过shell运行 (default: {False})
    """

    if shell:
        cmdstring_list = cmdstring
    else:
        cmdstring_list = shlex.split(cmdstring)
    if timeout:
        end_time = datetime.datetime.now() + datetime.timedelta(seconds=timeout)
    
    # 没有指定标准输出和错误输出的管道，因此会打印到屏幕上；
    sub = subprocess.Popen(cmdstring_list, cwd=cwd, stdin=subprocess.PIPE,shell=shell,bufsize=4096)
    
    # subprocess.poll()方法：检查子进程是否结束了，如果结束了，设定并返回码，放在subprocess.returncode变量中 
    while sub.poll() is None:
        time.sleep(0.1)
        if timeout:
            if end_time <= datetime.datetime.now():
                raise Exception("Timeout：%s"%cmdstring)
        
    return sub.returncode

def build_library():
    envs = os.environ
    cmake_prefix_path = envs.get('CMAKE_PREFIX_PATH', None)
    if cmake_prefix_path is None:
        raise EnvironmentError("Please specify CMAKE_PREFIX_PATH env.")
    
    config_command = "cmake -DCMAKE_PREFIX_PATH={} -S {} -B {}"
    path_to_source = cwd
    path_to_build = os.path.join(cwd, "build")
    
    if os.path.exists(path_to_build):
        shutil.rmtree(path_to_build)
    config_command = config_command.format(cmake_prefix_path, path_to_source, path_to_build)
    
    code = execute_command(config_command)
    if code != 0:
        raise RuntimeError("Run configure command fail.")
    
    build_command = "cmake --build {}".format(os.path.join(cwd, "build"))
    code = execute_command(build_command)
    if code != 0:
        raise RuntimeError("Run build Command fail.")

def main():
    build_library()
    extention = Extension(
        "matrix",
        libraries=["matrix"],
        sources=["stub.cc"],
        language="c++",
        extra_compile_args=['-std=c++11'],
        include_dirs=[cwd],
        library_dirs=[os.path.join(cwd, "build")]
    )
    setup(name="fputs",
          version="1.0.0",
          description="A Simple Matrix Library.",
          author="hanbing",
          author_email="beatmight@gmail.com",
          ext_modules=[extention]
          
         )

if __name__ == "__main__":
    main()