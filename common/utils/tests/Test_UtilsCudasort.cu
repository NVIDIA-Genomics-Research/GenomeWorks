/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../include/claragenomics/utils/cudasort.cuh"

namespace claragenomics
{
namespace cudautils
{
namespace cudasort
{
namespace tests
{

template <typename MoreSignificantKeyT,
          typename LessSignificantKeyT,
          typename ValueT>
void test_function(thrust::device_vector<MoreSignificantKeyT>& more_significant_keys,
                   thrust::device_vector<LessSignificantKeyT>& less_significant_keys,
                   thrust::device_vector<ValueT>& input_values)
{
    ASSERT_EQ(input_values.size(), more_significant_keys.size());
    ASSERT_EQ(input_values.size(), less_significant_keys.size());

    sort_by_two_keys(more_significant_keys,
                     less_significant_keys,
                     input_values);

    thrust::host_vector<ValueT> sorted_values_h(input_values);

    ASSERT_EQ(sorted_values_h.size(), input_values.size());
    // sort is done by two keys and not values, but tests cases are intentionally made so the values are sorted as well
    for (std::size_t i = 1; i < input_values.size(); ++i)
    {
        EXPECT_LE(sorted_values_h[i - 1], sorted_values_h[i]) << "index: " << i << std::endl;
    }
}

// repreat this test with differnt combinations of types
template <typename MoreSignificantKeyT,
          typename LessSignificantKeyT,
          typename ValueT>
void short_test_template()
{
    // more less value
    //    6    1     610
    //    2    4     240
    //    5    5     550
    //    4    2     420
    //    4    5     450
    //    2    1     210
    //    2    2     220
    //    3    8     380
    //    3    7     370
    //    5    1     510
    //    5    3     530
    //    4    5     451
    //    8    4     840
    const std::vector<MoreSignificantKeyT> more_significant_keys_vec = {6, 2, 5, 4, 4, 2, 2, 3, 3, 5, 5, 4, 8};
    const std::vector<LessSignificantKeyT> less_significant_keys_vec = {1, 4, 5, 2, 5, 1, 2, 8, 7, 1, 3, 5, 4};
    const std::vector<ValueT> input_values_vec                       = {610, 240, 550, 420, 450, 210, 220, 380, 370, 510, 530, 451, 840};

    thrust::device_vector<MoreSignificantKeyT> more_significant_keys(std::begin(more_significant_keys_vec),
                                                                     std::end(more_significant_keys_vec));
    thrust::device_vector<LessSignificantKeyT> less_significant_keys(std::begin(less_significant_keys_vec),
                                                                     std::end(less_significant_keys_vec));
    thrust::device_vector<ValueT> input_values(std::begin(input_values_vec),
                                               std::end(input_values_vec));

    test_function(more_significant_keys,
                  less_significant_keys,
                  input_values);
}

TEST(TestUtilsCudasort, short_32_32_32_test)
{
    short_test_template<std::uint32_t, std::uint32_t, std::uint32_t>();
}

TEST(TestUtilsCudasort, short_32_32_64_test)
{
    short_test_template<std::uint32_t, std::uint32_t, std::uint64_t>();
}

TEST(TestUtilsCudasort, short_32_64_32_test)
{
    short_test_template<std::uint32_t, std::uint64_t, std::uint32_t>();
}

TEST(TestUtilsCudasort, short_32_64_64_test)
{
    short_test_template<std::uint32_t, std::uint64_t, std::uint64_t>();
}

TEST(TestUtilsCudasort, short_64_32_32_test)
{
    short_test_template<std::uint64_t, std::uint32_t, std::uint32_t>();
}

TEST(TestUtilsCudasort, short_64_32_64_test)
{
    short_test_template<std::uint64_t, std::uint32_t, std::uint64_t>();
}

TEST(TestUtilsCudasort, short_64_64_32_test)
{
    short_test_template<std::uint64_t, std::uint64_t, std::uint32_t>();
}

TEST(TestUtilsCudasort, short_64_64_64_test)
{
    short_test_template<std::uint64_t, std::uint64_t, std::uint64_t>();
}

TEST(TestUtilsCudasort, long_deterministic_shuffle_test)
{
    std::size_t number_of_elements = 10'000'000;

    std::random_device rd;
    std::mt19937 g(rd());

    // fill the arrays with values 0..number_of_elements and shuffle them
    thrust::host_vector<std::uint32_t> more_significant_keys_h(number_of_elements);
    std::iota(std::begin(more_significant_keys_h), std::end(more_significant_keys_h), 0);
    std::shuffle(std::begin(more_significant_keys_h), std::end(more_significant_keys_h), g);
    thrust::device_vector<std::uint32_t> more_significant_keys_d(more_significant_keys_h);

    thrust::host_vector<std::uint32_t> less_significant_keys_h(number_of_elements);
    std::iota(std::begin(less_significant_keys_h), std::end(less_significant_keys_h), 0);
    std::shuffle(std::begin(less_significant_keys_h), std::end(less_significant_keys_h), g);
    thrust::device_vector<std::uint32_t> less_significant_keys_d(less_significant_keys_h);

    thrust::host_vector<std::uint64_t> input_values_h(number_of_elements);
    std::transform(std::begin(more_significant_keys_h),
                   std::end(more_significant_keys_h),
                   std::begin(less_significant_keys_h),
                   std::begin(input_values_h),
                   [number_of_elements](const std::uint32_t more_significant_key, const std::uint32_t less_significant_key) {
                       return number_of_elements * more_significant_key + less_significant_key;
                   });
    thrust::device_vector<std::uint64_t> input_values_d(input_values_h);

    test_function(more_significant_keys_d, less_significant_keys_d, input_values_d);
}

} //namespace tests
} //namespace cudasort
} //namespace cudautils
} //namespace claragenomics
