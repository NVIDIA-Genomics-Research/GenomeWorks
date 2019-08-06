/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <claragenomics/utils/cudautils.hpp>
#include <exception>

namespace claragenomics
{

class device_memory_allocation_exception : public std::exception
{
public:
    device_memory_allocation_exception()                                          = default;
    device_memory_allocation_exception(device_memory_allocation_exception const&) = default;
    device_memory_allocation_exception& operator=(device_memory_allocation_exception const&) = default;
    virtual ~device_memory_allocation_exception()                                            = default;

    virtual const char* what() const noexcept
    {
        return "Could not allocate device memory!";
    }
};

template <typename T>
class device_buffer
{
public:
    using value_type = T;
    device_buffer() = delete;
    device_buffer(size_t n_elements, int32_t device_id)
        : size_(n_elements)
        , device_id_(device_id)
    {
        CGA_CU_CHECK_ERR(cudaSetDevice(device_id_));
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T));
        if (err == cudaErrorMemoryAllocation)
            throw device_memory_allocation_exception();
        CGA_CU_CHECK_ERR(err);
    }

    ~device_buffer()
    {
        CGA_CU_CHECK_ERR(cudaSetDevice(device_id_));
        CGA_CU_CHECK_ERR(cudaFree(data_));
    }

    T* data() { return data_; }
    T const* data() const { return data_; }
    size_t size() const { return size_; }

private:
    T* data_;
    size_t size_;
    int32_t device_id_;
};

} // end namespace claragenomics
