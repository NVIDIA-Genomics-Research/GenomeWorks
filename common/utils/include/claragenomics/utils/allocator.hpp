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
    /// @brief Asynchronously allocates device memory.
    ////       An implementation of this need to return a allocation of n bytes properly aligned
    ///        on the configured device.
    /// @param n      number of bytes to allocate
    /// @param stream CUDA stream to be associated with this method.
    /// @returns a pointer to a n byte properly aligned device buffer on the configured device.
    virtual void* allocate(std::size_t n, cudaStream_t stream) = 0;

    /// @brief Asynchronously deallocates device memory.
    /// @param p      pointer to the buffer to deallocate
    /// @param n      size of the buffer to deallocate in bytes
    /// @param stream CUDA stream to be associated with this method.
    virtual void deallocate(void* p, std::size_t n, cudaStream_t stream) = 0;
};

/**
 * @brief Interface for a asynchronous host allocator.
 */
class HostAllocator
{
public:
    /// @brief Asynchronously allocates host memory.
    ////       An implementation of this need to return a allocation of n bytes properly aligned
    ///        on the host.
    /// @param n       number of bytes to allocate
    /// @param stream  CUDA stream to be associated with this method.
    /// @returns a pointer to a n byte properly aligned host buffer.
    virtual void* allocate(std::size_t n, cudaStream_t stream) = 0;

    /// @brief Asynchronously deallocates host memory.
    /// @param p      pointer to the buffer to deallocate
    /// @param n      size of the buffer to deallocate in bytes
    /// @param stream CUDA stream to be associated with this method.
    virtual void deallocate(void* p, std::size_t n, cudaStream_t stream) = 0;
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
    /// @brief This constructor intializes and constructs cub's CachingDeviceAllocator
    /// @param max_cached_bytes Maximum aggregate cached bytes per device (default is 1GB)
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
