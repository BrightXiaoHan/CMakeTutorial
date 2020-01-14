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
