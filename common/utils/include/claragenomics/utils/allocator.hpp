#pragma once
#include <cub/util_allocator.cuh>

namespace claragenomics
{

/** Default cudaMalloc/cudaFree based device allocator */
class deviceAllocator
{
public:
    virtual void* allocate(std::size_t n, cudaStream_t)
    {
        void* ptr = 0;
        CGA_CU_CHECK_ERR(cudaMalloc(&ptr, n));
        return ptr;
    }
    virtual void deallocate(void* p, std::size_t, cudaStream_t)
    {
        cudaError_t status = cudaFree(p);
        if (cudaSuccess != status)
        {
            // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        }
    }

    virtual ~deviceAllocator() {}
};

/** Default cudaHostMalloc/cudaFreeHost based host allocator */
class hostAllocator
{
public:
    virtual void* allocate(std::size_t n, cudaStream_t)
    {
        void* ptr = 0;
        CGA_CU_CHECK_ERR(cudaMallocHost(&ptr, n));
        return ptr;
    }
    virtual void deallocate(void* p, std::size_t, cudaStream_t)
    {
        cudaError_t status = cudaFreeHost(p);
        if (cudaSuccess != status)
        {
            // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        }
    }

    virtual ~hostAllocator() {}
};

class cachingDeviceAllocator : public deviceAllocator
{
public:
    cachingDeviceAllocator()
        : _allocator(8, 3, cub::CachingDeviceAllocator::INVALID_BIN, cub::CachingDeviceAllocator::INVALID_SIZE, false, false)
    {
    }

    virtual void* allocate(std::size_t n, cudaStream_t stream)
    {
        void* ptr = 0;
        _allocator.DeviceAllocate(&ptr, n, stream);
        return ptr;
    }

    virtual void deallocate(void* p, std::size_t, cudaStream_t)
    {
        // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        _allocator.DeviceFree(p);
    }

private:
    cub::CachingDeviceAllocator _allocator;
};

} // namespace claragenomics
