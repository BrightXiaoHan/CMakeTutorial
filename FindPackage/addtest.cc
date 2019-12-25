#include <iostream>
#include "libadd.h"

int main(int argc, char const *argv[])
{
    int a = 2;
    int b = 3;
    int result;
    result = add(a, b);
    std::cout << a << "加" << b << "等于" <<result <<std::endl;
    return 0;
}

