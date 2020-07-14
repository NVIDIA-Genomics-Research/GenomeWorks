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

#include <claraparabricks/genomeworks/gw_config.hpp>
#include <limits>
#include <cstdint>

namespace claraparabricks
{

namespace genomeworks
{
#ifdef GW_CUDA_BEFORE_10_1
template <typename T>
struct numeric_limits
{
};

template <>
struct numeric_limits<int16_t>
{
    GW_CONSTEXPR static __device__ int16_t max() { return INT16_MAX; }
    GW_CONSTEXPR static __device__ int16_t min() { return INT16_MIN; }
};

template <>
struct numeric_limits<int32_t>
{
    GW_CONSTEXPR static __device__ int32_t max() { return INT32_MAX; }
    GW_CONSTEXPR static __device__ int32_t min() { return INT32_MIN; }
};
#else
using std::numeric_limits;
#endif

} // namespace genomeworks

} // namespace claraparabricks
