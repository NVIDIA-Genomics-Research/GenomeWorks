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
/// \def GW_CU_CHECK_ERR
/// \brief Log on CUDA error in enclosed expression
#define GW_CU_CHECK_ERR(ans)                                          \
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
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        GW_LOG_ERROR("GPU Error:: {} {} {}\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

} // namespace cudautils

} // namespace genomeworks