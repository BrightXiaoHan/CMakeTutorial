#include "dev_array.h"

// set
template <class T> 
void dev_array<T>::set(const T* src, size_t size)
{
    size_t min = std::min(size, getSize());
    cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to device memory");
    }
}

// get
template <class T>
void dev_array<T>::get(T* dest, size_t size)
{
    size_t min = std::min(size, getSize());
    cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to host memory");
    }
}

// allocate memory on the device
template <class T>
void dev_array<T>::allocate(size_t size)
{
    cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
    if (result != cudaSuccess)
    {
        start_ = end_ = 0;
        throw std::runtime_error("failed to allocate device memory");
    }
    end_ = start_ + size;
}

// free memory on the device
template <class T>
void dev_array<T>::free()
{
    if (start_ != 0)
    {
        cudaFree(start_);
        start_ = end_ = 0;
    }
}