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
/// \file
/// \defgroup cudautils Internal CUDA utilities package

#include <cuda_runtime_api.h>
#include <logging/logging.hpp>

/// \ingroup cudautils
/// \{

/// \ingroup cudautils
/// \def CGA_CU_CHECK_ERR
/// \brief Log on CUDA error in enclosed expression
#define CGA_CU_CHECK_ERR(ans)                                           \
    {                                                                   \
        claragenomics::cudautils::gpuAssert((ans), __FILE__, __LINE__); \
    }

/// \}

namespace claragenomics
{

namespace cudautils
{

/// gpuAssert
/// Logs and/or exits on cuda error
/// \ingroup cudautils
/// \param code The CUDA status code of the function being asserted
/// \param file Filename of the calling function
/// \param line File line number of the calling function
/// \param abort If true, hard-exit on CUDA error
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
    if (code != cudaSuccess)
    {
        std::string err = "GPU Error:: " +
                          std::string(cudaGetErrorString(code)) +
                          " " + std::string(file) +
                          " " + std::to_string(line);
        if (abort)
        {
            CGA_LOG_ERROR("{}\n", err);
            std::abort();
        }
        else
        {
            throw std::runtime_error(err);
        }
    }

}

/// make_unique_cuda_malloc
/// Creates a unique pointer to device memory which gets deallocated automatically
/// \param num_of_elems number of elements to allocate
/// \return unique pointer to allocated memory
template<typename T>
std::unique_ptr<T, void(*)(T*)> make_unique_cuda_malloc(std::size_t num_of_elems) {
    T* tmp_ptr_d = nullptr;
    CGA_CU_CHECK_ERR(cudaMalloc((void**)&tmp_ptr_d, num_of_elems*sizeof(T)));
    std::unique_ptr<T, void(*)(T*)> unq_ptr_d(tmp_ptr_d, [](T* p) {CGA_CU_CHECK_ERR(cudaFree(p));}); // tmp_prt_d's ownership transfered to unq_ptr_d
    return std::move(unq_ptr_d);
}

/// make_shared_cuda_malloc
/// Creates a shared pointer to device memory which gets deallocated automatically
/// \param num_of_elems number of elements to allocate
/// \return shared pointer to allocated memory
template<typename T>
std::shared_ptr<T> make_shared_cuda_malloc(std::size_t num_of_elems) {
    T* tmp_ptr_d = nullptr;
    CGA_CU_CHECK_ERR(cudaMalloc((void**)&tmp_ptr_d, num_of_elems*sizeof(T)));
    std::shared_ptr<T> shr_ptr_d(tmp_ptr_d, [](T* p) {CGA_CU_CHECK_ERR(cudaFree(p));}); // tmp_prt_d's ownership transfered to shr_ptr_d
    return std::move(shr_ptr_d);
}

} // namespace cudautils

} // namespace claragenomics
