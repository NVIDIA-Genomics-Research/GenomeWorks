/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#ifdef __CUDACC_VER_MAJOR__
/// \def GW_CUDA_BEFORE_XX_X
/// \brief Macros to enable/disable CUDA version specific code
#define GW_CUDA_BEFORE_XX_X 1

#if __CUDACC_VER_MAJOR__ < 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ < 1)
#define GW_CUDA_BEFORE_10_1
#endif

#if (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ < 2)
#define GW_CUDA_BEFORE_9_2
#endif
#endif

/// \def GW_CONSTEXPR
/// \brief C++ constexpr for device code - falls back to const for CUDA 10.0 and earlier
#ifdef GW_CUDA_BEFORE_10_1
#define GW_CONSTEXPR const
#else
#define GW_CONSTEXPR constexpr
#endif
