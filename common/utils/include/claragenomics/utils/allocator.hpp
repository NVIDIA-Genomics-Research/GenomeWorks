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
#include <type_traits>

#include <cub/util_allocator.cuh>
#include <claragenomics/utils/device_preallocated_allocator.cuh>

#include <claragenomics/utils/cudautils.hpp>

namespace claragenomics
{

/// @brief Allocator that allocates device memory using cudaMalloc/cudaFree
template <typename T>
class CudaMallocAllocator
{
public:
    /// type of elements of allocated array
    using value_type = T;

    /// pointer to elements of allocated array
    using pointer = T*;

    /// @brief default constructor
    CudaMallocAllocator() = default;

    /// @brief copy constructor
    /// @param rhs input allocator
    CudaMallocAllocator(const CudaMallocAllocator& rhs) = default;

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
    CudaMallocAllocator& operator=(const CudaMallocAllocator& rhs) = default;

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
    CudaMallocAllocator(CudaMallocAllocator&& rhs) = default;

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
    CudaMallocAllocator& operator=(CudaMallocAllocator&& rhs) = default;

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
    ~CudaMallocAllocator() = default;

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

/// @brief A simple caching allocator for device memory allocations
/// @tparam T
/// @tparam MemoryResource resource that does actual allocation, e.g. cub::CachingDeviceAllocator or cudautils::DevicePreallocatedAllocator
template <typename T, typename MemoryResource>
class CachingDeviceAllocator
{
public:
    /// type of elements of allocated array
    using value_type = T;

    /// pointer to elements of allocated array
    using pointer = T*;

    /// @brief Constructor
    /// @param max_cached_bytes max bytes used by memory resource (default is 1GiB)
    CachingDeviceAllocator(size_t max_cached_bytes = 1024*1024*1024)
        : memory_resource_(generate_memory_resource(max_cached_bytes))
    {
    }

    /// @brief copy constructor
    /// @param rhs input allocator
    CachingDeviceAllocator(const CachingDeviceAllocator& rhs) = default;

    /// @brief copy constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    template <typename U>
    CachingDeviceAllocator(const CachingDeviceAllocator<U, MemoryResource>& rhs)
        : memory_resource_(rhs.memory_resource())
    {
    }

    /// @brief copy assignment operator
    /// @param rhs input allocator
    /// @return reference to this object
    CachingDeviceAllocator& operator=(const CachingDeviceAllocator& rhs) = default;

    /// @brief copy assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    /// @return reference to this object
    template <typename U>
    CachingDeviceAllocator& operator=(const CachingDeviceAllocator<U, MemoryResource>& rhs)
    {
        memory_resource_ = rhs.memory_resource();
        return *this;
    }

    /// @brief move constructor
    /// @param rhs input allocator
    CachingDeviceAllocator(CachingDeviceAllocator&& rhs) = default;

    /// @brief move constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    template <typename U>
    CachingDeviceAllocator(CachingDeviceAllocator<U, MemoryResource>&& rhs)
        : memory_resource_(rhs.memory_resource())
    {
    }

    /// @brief move assignment operator
    /// @param rhs input allocator
    /// @return reference to this object
    CachingDeviceAllocator& operator=(CachingDeviceAllocator&& rhs) = default;

    /// @brief move assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// @param rhs input allocator
    /// @tparam U Type of rhs::value_type
    /// @return reference to this object
    template <typename U>
    CachingDeviceAllocator& operator=(CachingDeviceAllocator<U, MemoryResource>&& rhs)
    {
        memory_resource_ = rhs.memory_resource();
        return *this;
    }

    /// @brief destructor
    ~CachingDeviceAllocator() = default;

    /// @brief asynchronously allocates a device array with enough space for n elements of value_type
    /// @param n number of elements to allocate the array for
    /// @param stream CUDA stream to be associated with this method
    /// @return pointer to allocated array
    pointer allocate(std::size_t n, cudaStream_t stream = 0)
    {
        void* ptr = 0;
        CGA_CU_CHECK_ERR(memory_resource_->DeviceAllocate(&ptr, n * sizeof(T), stream));
        return static_cast<pointer>(ptr);
    }

    /// @brief Asynchronously dealllocates allocated array
    /// @param p pointer to the array to deallocate
    /// @param n number of elements the array was allocated for
    /// @param stream CUDA stream to be associated with this method.
    void deallocate(pointer p, std::size_t n, cudaStream_t stream = 0)
    {
        // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        CGA_CU_ABORT_ON_ERR(memory_resource_->DeviceFree(p));
    }

    /// @brief returns a shared pointer to memory_resource
    /// @return a shared pointer to memory_resource
    std::shared_ptr<MemoryResource> memory_resource() const { return memory_resource_; }

private:
    /// @brief Special constructor for cub::CachingDeviceAllocator
    /// @param max_cached_bytes
    /// @tparam is_cub_allocator true
    /// @return memory resource
    template <bool is_cub_allocator = std::is_same<MemoryResource, cub::CachingDeviceAllocator>::value>
    std::enable_if_t<is_cub_allocator, std::shared_ptr<MemoryResource>>
    generate_memory_resource(size_t max_cached_bytes)
    {
        /// Smallest cached bin is 2^10 bytes, largest is 2^28 bytes. All allocation requests larger than 2^28 bytes are not fit in a bin and are not cached
        return std::make_shared<cub::CachingDeviceAllocator>(2, 10, 28, max_cached_bytes, false, false);
    }

    /// @brief Constructor for all other memory resources
    /// @param max_cached_bytes
    /// @tparam is_cub_allocator false
    /// @return memory resource
    template <bool is_cub_allocator = std::is_same<MemoryResource, cub::CachingDeviceAllocator>::value>
    std::enable_if_t<!is_cub_allocator, std::shared_ptr<MemoryResource>>
    generate_memory_resource(size_t max_cached_bytes)
    {
        return std::make_shared<MemoryResource>(max_cached_bytes);
    }

    std::shared_ptr<MemoryResource> memory_resource_;
};

//#ifdef CGA_ENABLE_ALLOCATOR
/// Default device allocator do be used if CGA_ENABLE_ALLOCATOR is set
//using DefaultDeviceAllocator = CachingDeviceAllocator<char, cub::CachingDeviceAllocator>;
using DefaultDeviceAllocator = CachingDeviceAllocator<char, DevicePreallocatedAllocator>;
//#else
/// Default device allocator do be used if CGA_ENABLE_ALLOCATOR is not set
//using DefaultDeviceAllocator = CudaMallocAllocator<char>;
//#endif

} // namespace claragenomics
