#include <iostream>
#include "glog/logging.h"

int main(int argc, char const *argv[])
{
    google::InitGoogleLogging(argv[0]);    // 初始化
    double input, result;
    LOG(INFO) << "测试glog";
    std::cout << "测试glog成功" << std::endl;
    
    return 0;
}
