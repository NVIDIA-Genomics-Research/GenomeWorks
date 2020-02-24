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

#include <memory>

#include <cub/util_allocator.cuh>

#include <claragenomics/utils/cudautils.hpp>

namespace claragenomics
{

/**
 * @brief Allocator that allocates device memory using cudaMalloc/cudaFree
 */
template <typename T>
class CudaMallocAllocator
{
public:
    /// type of elements of allocated array
    using value_type = T;

    /// pointer to elements of allocated array
    using pointer = T*;

    /// @brief default constructor
    CudaMallocAllocator(){};

    /// @brief copy constructor
    /// @param rhs input allocator
    CudaMallocAllocator(const CudaMallocAllocator& rhs) {}

    /// @brief copy constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    template <typename U>
    CudaMallocAllocator(const CudaMallocAllocator<U>& rhs)
    {
    }

    /// @brief copy assignment operator
    /// @param rhs input allocator
    /// @return reference to this object
    CudaMallocAllocator& operator=(const CudaMallocAllocator& rhs) { return *this; }

    /// @brief copy assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    /// @return reference to this object
    template <typename U>
    CudaMallocAllocator& operator=(const CudaMallocAllocator<U>& rhs)
    {
        return *this;
    }

    /// @brief move constructor
    /// @param rhs input allocator
    CudaMallocAllocator(CudaMallocAllocator&& rhs) {}

    /// @brief move constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    template <typename U>
    CudaMallocAllocator(CudaMallocAllocator<U>&& rhs)
    {
    }

    /// @brief move assignment operator
    /// @param rhs input allocator
    /// @return reference to this object
    CudaMallocAllocator& operator=(CudaMallocAllocator&& rhs) { return *this; }

    /// @brief move assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    /// @return reference to this object
    template <typename U>
    CudaMallocAllocator& operator=(CudaMallocAllocator<U>&& rhs)
    {
        return *this;
    }

    /// @brief destructor
    virtual ~CudaMallocAllocator() {}

    /// @brief asynchronously allocates a device array with enough space for n elements of value_type
    /// @param n number of elements to allocate the array for
    /// @param stream CUDA stream to be associated with this method
    /// @return pointer to allocated array
    pointer allocate(std::size_t n, cudaStream_t stream = 0)
    {
        void* ptr = 0;
        CGA_CU_CHECK_ERR(cudaMalloc(&ptr, n * sizeof(T)));
        return static_cast<pointer>(ptr);
    }

    /// @brief Asynchronously dealllocates allocated array
    /// @param p pointer to the array to deallocate
    /// @param n number of elements the array was allocated for
    /// @param stream CUDA stream to be associated with this method.
    void deallocate(pointer p, std::size_t n, cudaStream_t stream = 0)
    {
        cudaError_t status = cudaFree(p);
        if (cudaSuccess != status)
        {
            // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        }
    }
};

/**
 * @brief A simple caching allocator for device memory allocations
 */
template <typename T>
class CachingDeviceAllocator
{
public:
    /// type of elements of allocated array
    using value_type = T;

    /// pointer to elements of allocated array
    using pointer = T*;

    /// @brief This constructor intializes and constructs cub's CachingDeviceAllocator
    /// Smallest cached bin is 2^10 bytes, largest is 2^28 bytes. All allocation requests larger than 2^28 bytes are not fit in a bin and are not cached
    /// @param max_cached_bytes Maximum aggregate cached bytes per device (default is 1GB)
    CachingDeviceAllocator(size_t max_cached_bytes = 1e9)
        : cub_allocator_(std::make_shared<cub::CachingDeviceAllocator>(2, 10, 28, max_cached_bytes, false, false))
    {
    }

    /// @brief copy constructor
    /// @param rhs input allocator
    CachingDeviceAllocator(const CachingDeviceAllocator& rhs)
        : cub_allocator_(rhs.cub_allocator_)
    {
    }

    /// @brief copy constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    template <typename U>
    CachingDeviceAllocator(const CachingDeviceAllocator<U>& rhs)
        : cub_allocator_(rhs.cub_allocator())
    {
    }

    /// @brief copy assignment operator
    /// @param rhs input allocator
    /// @return reference to this object
    CachingDeviceAllocator& operator=(const CachingDeviceAllocator& rhs)
    {
        cub_allocator_ = rhs.cub_allocator_;
        return *this;
    }

    /// @brief copy assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    /// @return reference to this object
    template <typename U>
    CachingDeviceAllocator& operator=(const CachingDeviceAllocator<U>& rhs)
    {
        cub_allocator_ = rhs.cub_allocator();
        return *this;
    }

    /// @brief move constructor
    /// @param rhs input allocator
    CachingDeviceAllocator(CachingDeviceAllocator&& rhs)
        : cub_allocator_(rhs.cub_allocator_)
    {
    }

    /// @brief move constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    template <typename U>
    CachingDeviceAllocator(CachingDeviceAllocator<U>&& rhs)
        : cub_allocator_(rhs.cub_allocator())
    {
    }

    /// @brief move assignment operator
    /// @param rhs input allocator
    /// @return reference to this object
    CachingDeviceAllocator& operator=(CachingDeviceAllocator&& rhs)
    {
        cub_allocator_ = rhs.cub_allocator_;
        return *this;
    }

    /// @brief move assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    /// @return reference to this object
    template <typename U>
    CachingDeviceAllocator& operator=(CachingDeviceAllocator<U>&& rhs)
    {
        cub_allocator_ = rhs.cub_allocator();
        return *this;
    }

    /// @brief destructor
    virtual ~CachingDeviceAllocator()
    {
        // no need to explicitly clear memory as long as cub::CachingDeviceAllocator allocator is called with skip_cleanup = false
    }

    /// @brief asynchronously allocates a device array with enough space for n elements of value_type
    /// @param n number of elements to allocate the array for
    /// @param stream CUDA stream to be associated with this method
    /// @return pointer to allocated array
    pointer allocate(std::size_t n, cudaStream_t stream = 0)
    {
        void* ptr = 0;
        cub_allocator_->DeviceAllocate(&ptr, n * sizeof(T), stream);
        return static_cast<pointer>(ptr);
    }

    /// @brief Asynchronously dealllocates allocated array
    /// @param p pointer to the array to deallocate
    /// @param n number of elements the array was allocated for
    /// @param stream CUDA stream to be associated with this method.
    void deallocate(pointer p, std::size_t n, cudaStream_t stream = 0)
    {
        // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        cub_allocator_->DeviceFree(p);
    }

    /// @brief returns a shared pointer to internally used cub::CachingDeviceAllocator
    /// @return a shared pointer to internally used cub::CachingDeviceAllocator
    std::shared_ptr<cub::CachingDeviceAllocator> cub_allocator() const { return cub_allocator_; }

private:
    std::shared_ptr<cub::CachingDeviceAllocator> cub_allocator_;
};

#ifdef CGA_ENABLE_ALLOCATOR
/// Default device allocator do be used if CGA_ENABLE_ALLOCATOR is set
using DefaultDeviceAllocator = CachingDeviceAllocator<char>;
#else
/// Default device allocator do be used if CGA_ENABLE_ALLOCATOR is not set
using DefaultDeviceAllocator = CudaMallocAllocator<char>;
#endif

} // namespace claragenomics
