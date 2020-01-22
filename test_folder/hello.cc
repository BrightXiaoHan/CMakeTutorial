#include "hello.h"
#include <iostream>

void MyClass::printValue(){
    std::cout << value << std::endl;
}


int main(int argc, char const *argv[])
{
    MyClass obj;
    obj.printValue();
    return 0;
}


