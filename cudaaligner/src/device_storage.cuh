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

#include <cudautils/cudautils.hpp>

namespace claragenomics
{

template <typename T>
class device_storage
{
public:
    using value_type = T;
    device_storage() = delete;
    device_storage(size_t n_elements, int32_t device_id)
        : size_(n_elements)
        , device_id_(device_id)
    {
        CGA_CU_CHECK_ERR(cudaSetDevice(device_id_));
        CGA_CU_CHECK_ERR(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(T)));
    }

    ~device_storage()
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
