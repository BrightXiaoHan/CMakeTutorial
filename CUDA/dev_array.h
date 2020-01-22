#ifndef _DEV_ARRAY_H_
#define _DEV_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class dev_array
{
// public functions
public:
    explicit dev_array()
        : start_(0),
          end_(0)
    {}

    // constructor
    explicit dev_array(size_t size)
    {
        allocate(size);
    }
    // destructor
    ~dev_array()
    {
        free();
    }

    // resize the vector
    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    // get the size of the array
    size_t getSize() const
    {
        return end_ - start_;
    }

    // get data
    const T* getData() const
    {
        return start_;
    }

    T* getData()
    {
        return start_;
    }

    // set
    void set(const T* src, size_t size);
    // get
    void get(T* dest, size_t size);

// private functions
private:
    // allocate memory on the device
    void allocate(size_t size);

    // free memory on the device
    void free();

    T* start_;
    T* end_;
};

#endif