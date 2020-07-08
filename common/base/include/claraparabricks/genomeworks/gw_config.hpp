

#pragma once
/// \def GW_CUDA_BEFORE_XX_X
/// \brief Macros to enable/disable CUDA version specific code
#define GW_CUDA_BEFORE_XX_X 1

#if __CUDACC_VER_MAJOR__ < 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ < 1)
#define GW_CUDA_BEFORE_10_1
#endif

#if (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ < 2)
#define GW_CUDA_BEFORE_9_2
#endif

/// \def GW_CONSTEXPR
/// \brief C++ constexpr for device code - falls back to const for CUDA 10.0 and earlier
#ifdef GW_CUDA_BEFORE_10_1
#define GW_CONSTEXPR const
#else
#define GW_CONSTEXPR constexpr
#endif
