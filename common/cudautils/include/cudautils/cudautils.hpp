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
#define CGA_CU_CHECK_ERR(ans)                                          \
    {                                                                 \
        genomeworks::cudautils::gpuAssert((ans), __FILE__, __LINE__); \
    }

/// \}

namespace genomeworks
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

} // namespace cudautils

} // namespace genomeworks
