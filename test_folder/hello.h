#ifndef _MYCLASS_H
#define _MYCLASS_H

class MyClass{
    public:
        explicit MyClass()
            :value(0)
        {};

        void printValue();

    private:
        double value;
};

#endif