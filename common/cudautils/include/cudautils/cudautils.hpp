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
#define GW_CU_CHECK_ERR(ans) { genomeworks::cudautils::gpuAssert((ans), __FILE__, __LINE__); }

/// \}

namespace genomeworks {

    namespace cudautils {

        inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
        {
            if (code != cudaSuccess)
            {
                GW_LOG_ERROR("GPU Error:: {} {} {}\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
            }
        }

    }

}