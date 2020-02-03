/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once
#include <cub/util_allocator.cuh>

namespace claragenomics
{

/**
 * @brief Interface for a asynchronous device allocator.
 */
class DeviceAllocator
{
public:
    virtual void* allocate(std::size_t n, cudaStream_t stream) = 0;

    virtual void deallocate(void* p, std::size_t, cudaStream_t stream) = 0;
};

/**
 * @brief Interface for a asynchronous host allocator.
 */
class HostAllocator
{
public:
    virtual void* allocate(std::size_t n, cudaStream_t stream) = 0;

    virtual void deallocate(void* p, std::size_t, cudaStream_t strean) = 0;
};

/**
 * @brief Default cudaMalloc/cudaFree based device allocator 
 */
class CudaMallocAllocator : public DeviceAllocator
{
public:
    void* allocate(std::size_t n, cudaStream_t) override
    {
        void* ptr = 0;
        CGA_CU_CHECK_ERR(cudaMalloc(&ptr, n));
        return ptr;
    }
    void deallocate(void* p, std::size_t, cudaStream_t) override
    {
        cudaError_t status = cudaFree(p);
        if (cudaSuccess != status)
        {
            // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        }
    }
};

/**
 * @brief A simple caching allocator for device memory allocations.
 */
class CachingDeviceAllocator : public DeviceAllocator
{
public:
    CachingDeviceAllocator(size_t max_cached_bytes = 1e9)
        : _allocator(2, 10, cub::CachingDeviceAllocator::INVALID_BIN, max_cached_bytes, false, false)
    {
    }

    void* allocate(std::size_t n, cudaStream_t stream) override
    {
        void* ptr = 0;
        _allocator.DeviceAllocate(&ptr, n, stream);
        return ptr;
    }

    void deallocate(void* p, std::size_t, cudaStream_t) override
    {
        // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        _allocator.DeviceFree(p);
    }

private:
    cub::CachingDeviceAllocator _allocator;
};

} // namespace claragenomics
