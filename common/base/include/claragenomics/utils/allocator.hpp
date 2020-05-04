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

#include <cuda_runtime_api.h>
#include <claragenomics/utils/device_preallocated_allocator.cuh>

#include <claragenomics/logging/logging.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/exceptions.hpp>

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
        static_cast<void>(rhs);
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
        static_cast<void>(stream);
        void* ptr       = nullptr;
        cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));
        if (err == cudaErrorMemoryAllocation)
        {
            // Clear the error from the runtime...
            err = cudaGetLastError();
            // Did a different (async) error happen in the meantime?
            if (err != cudaErrorMemoryAllocation)
            {
                CGA_CU_CHECK_ERR(err);
            }
            throw device_memory_allocation_exception();
        }
        CGA_CU_CHECK_ERR(err);
        return static_cast<pointer>(ptr);
    }

    /// @brief Asynchronously dealllocates allocated array
    /// @param p pointer to the array to deallocate
    /// @param n number of elements the array was allocated for
    /// @param stream CUDA stream to be associated with this method.
    void deallocate(pointer p, std::size_t n, cudaStream_t stream = 0)
    {
        static_cast<void>(n);
        static_cast<void>(stream);
        CGA_CU_ABORT_ON_ERR(cudaFree(p));
    }
};

/// @brief A simple caching allocator for device memory allocations
/// @tparam T
/// @tparam MemoryResource resource that does actual allocation, e.g. cudautils::DevicePreallocatedAllocator
template <typename T, typename MemoryResource>
class CachingDeviceAllocator
{
public:
    /// type of elements of allocated array
    using value_type = T;

    /// pointer to elements of allocated array
    using pointer = T*;

    /// @brief Default constructor
    /// Constructs an invalid CachingDeviceAllocator to allow default-construction of containers.
    /// A container using this allocator needs obtain a non-default constructed CachingDeviceAllocator object before performing any allocations.
    /// This can be achieved through through container assignment for example.
    CachingDeviceAllocator() = default;

    /// @brief Constructor
    /// @param max_cached_bytes max bytes used by memory resource
    CachingDeviceAllocator(size_t max_cached_bytes)
        : memory_resource_(std::make_shared<MemoryResource>(max_cached_bytes))
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
        if (!memory_resource_)
        {
            CGA_LOG_ERROR("{}\n", "ERROR:: Trying to allocate memory from an default-constructed CachingDeviceAllocator. Please assign a non-default-constructed CachingDeviceAllocator before performing any memory operations.");
            assert(false);
            std::abort();
        }
        void* ptr       = nullptr;
        cudaError_t err = memory_resource_->DeviceAllocate(&ptr, n * sizeof(T), stream);
        if (err == cudaErrorMemoryAllocation)
        {
            throw device_memory_allocation_exception();
        }
        CGA_CU_CHECK_ERR(err);
        return static_cast<pointer>(ptr);
    }

    /// @brief Asynchronously dealllocates allocated array
    /// @param p pointer to the array to deallocate
    /// @param n number of elements the array was allocated for
    /// @param stream CUDA stream to be associated with this method.
    void deallocate(pointer p, std::size_t n, cudaStream_t stream = 0)
    {
        static_cast<void>(n);
        static_cast<void>(stream);
        if (!memory_resource_)
        {
            CGA_LOG_ERROR("{}\n", "ERROR:: Trying to deallocate memory from an default-constructed CachingDeviceAllocator. Please assign a non-default-constructed CachingDeviceAllocator before performing any memory operations.");
            assert(false);
            std::abort();
        }
        // deallocate should not throw execeptions which is why CGA_CU_CHECK_ERR is not used.
        CGA_CU_ABORT_ON_ERR(memory_resource_->DeviceFree(p));
    }

    /// @brief returns a shared pointer to memory_resource
    /// @return a shared pointer to memory_resource
    std::shared_ptr<MemoryResource> memory_resource() const { return memory_resource_; }

private:
    std::shared_ptr<MemoryResource> memory_resource_;
};

#ifdef CGA_ENABLE_CACHING_ALLOCATOR
using DefaultDeviceAllocator = CachingDeviceAllocator<char, DevicePreallocatedAllocator>;
#else
using DefaultDeviceAllocator = CudaMallocAllocator<char>;
#endif

/// Constructs a DefaultDeviceAllocator
///
/// This function provides a way to construct a valid DefaultDeviceAllocator
/// for all possible DefaultDeviceAllocators.
/// Use this function to obtain a DefaultDeviceAllocator object.
/// This function is needed, since construction of CachingDeviceAllocator
/// requires a max_caching_size argument to obtain a valid allocator.
/// Default constuction of CachingDeviceAllocator yields an dummy object
/// which cannot allocate memory.
/// @param max_cached_bytes max bytes used by memory resource used by CachingDeviceAllocator (default: 2GiB, unused for CudaMallocAllocator)
inline DefaultDeviceAllocator create_default_device_allocator(std::size_t max_caching_size = 2ull * 1024 * 1024 * 1024)
{
#ifdef CGA_ENABLE_CACHING_ALLOCATOR
    return DefaultDeviceAllocator(max_caching_size);
#else
    static_cast<void>(max_caching_size);
    return DefaultDeviceAllocator();
#endif
}

} // namespace claragenomics
