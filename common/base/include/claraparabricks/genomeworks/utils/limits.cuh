

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
