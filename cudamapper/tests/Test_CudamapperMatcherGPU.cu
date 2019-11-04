/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

#include "../src/matcher_gpu.cuh"

namespace claragenomics
{

namespace cudamapper
{

void test_create_new_value_mask(const thrust::host_vector<representation_t>& representations_h,
                                const thrust::host_vector<std::uint32_t>& expected_new_value_mask_h,
                                std::uint32_t number_of_threads)
{
    const thrust::device_vector<representation_t> representations_d(representations_h);
    thrust::device_vector<std::uint32_t> new_value_mask_d(representations_h.size());

    std::uint32_t number_of_blocks = (representations_h.size() - 1) / number_of_threads + 1;

    details::matcher_gpu::create_new_value_mask<<<number_of_blocks, number_of_threads>>>(thrust::raw_pointer_cast(representations_d.data()),
                                                                                         representations_d.size(),
                                                                                         thrust::raw_pointer_cast(new_value_mask_d.data()));

    CGA_CU_CHECK_ERR(cudaDeviceSynchronize());

    const thrust::host_vector<std::uint32_t> new_value_mask_h(new_value_mask_d);

    ASSERT_EQ(new_value_mask_h.size(), expected_new_value_mask_h.size());
    for (std::size_t i = 0; i < expected_new_value_mask_h.size(); ++i)
    {
        EXPECT_EQ(new_value_mask_h[i], expected_new_value_mask_h[i]) << "index: " << i;
    }
}

TEST(TestCudamapperMatcherGPU, test_create_new_value_mask_small_example)
{
    thrust::host_vector<representation_t> representations_h;
    thrust::host_vector<std::uint32_t> expected_new_value_mask_h;
    representations_h.push_back(0);
    expected_new_value_mask_h.push_back(1);
    representations_h.push_back(0);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(0);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(0);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(0);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(3);
    expected_new_value_mask_h.push_back(1);
    representations_h.push_back(3);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(3);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(4);
    expected_new_value_mask_h.push_back(1);
    representations_h.push_back(5);
    expected_new_value_mask_h.push_back(1);
    representations_h.push_back(5);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(8);
    expected_new_value_mask_h.push_back(1);
    representations_h.push_back(8);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(8);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(9);
    expected_new_value_mask_h.push_back(1);
    representations_h.push_back(9);
    expected_new_value_mask_h.push_back(0);
    representations_h.push_back(9);
    expected_new_value_mask_h.push_back(0);

    std::uint32_t number_of_threads = 3;

    test_create_new_value_mask(representations_h,
                               expected_new_value_mask_h,
                               number_of_threads);
}

TEST(TestCudamapperMatcherGPU, test_create_new_value_mask_small_data_large_example)
{
    const std::uint64_t total_sketch_elements                    = 10000000;
    const std::uint32_t sketch_elements_with_same_representation = 1000;

    thrust::host_vector<representation_t> representations_h;
    thrust::host_vector<std::uint32_t> expected_new_value_mask_h;
    for (std::size_t i = 0; i < total_sketch_elements; ++i)
    {
        representations_h.push_back(i / sketch_elements_with_same_representation);
        if (i % sketch_elements_with_same_representation == 0)
            expected_new_value_mask_h.push_back(1);
        else
            expected_new_value_mask_h.push_back(0);
    }

    std::uint32_t number_of_threads = 256;

    test_create_new_value_mask(representations_h,
                               expected_new_value_mask_h,
                               number_of_threads);
}
void test_copy_index_of_first_occurence(const thrust::host_vector<std::uint64_t>& representation_index_mask_h,
                                        const thrust::host_vector<std::uint32_t>& expected_starting_index_of_each_representation_h,
                                        const std::uint32_t number_of_threads)
{
    const thrust::device_vector<std::uint64_t> representation_index_mask_d(representation_index_mask_h);
    ASSERT_EQ(expected_starting_index_of_each_representation_h.size(), representation_index_mask_h.back());
    thrust::device_vector<std::uint32_t> starting_index_of_each_representation_d(expected_starting_index_of_each_representation_h.size());

    std::uint32_t number_of_blocks = (representation_index_mask_d.size() - 1) / number_of_threads + 1;

    details::matcher_gpu::copy_index_of_first_occurence<<<number_of_blocks, number_of_threads>>>(thrust::raw_pointer_cast(representation_index_mask_d.data()),
                                                                                                 representation_index_mask_d.size(),
                                                                                                 thrust::raw_pointer_cast(starting_index_of_each_representation_d.data()));
    CGA_CU_CHECK_ERR(cudaDeviceSynchronize());

    const thrust::host_vector<std::uint32_t> starting_index_of_each_representation_h(starting_index_of_each_representation_d);

    ASSERT_EQ(starting_index_of_each_representation_h.size(), expected_starting_index_of_each_representation_h.size());
    for (std::size_t i = 0; i < expected_starting_index_of_each_representation_h.size(); ++i)
    {
        EXPECT_EQ(starting_index_of_each_representation_h[i], expected_starting_index_of_each_representation_h[i]) << "index: " << i;
    }
}

TEST(TestCudamapperMatcherGPU, test_copy_index_of_first_occurence_small_example)
{
    thrust::host_vector<std::uint64_t> representation_index_mask_h;
    thrust::host_vector<std::uint32_t> expected_starting_index_of_each_representation_h;
    representation_index_mask_h.push_back(1);
    expected_starting_index_of_each_representation_h.push_back(0);
    representation_index_mask_h.push_back(1);
    representation_index_mask_h.push_back(1);
    representation_index_mask_h.push_back(1);
    representation_index_mask_h.push_back(2);
    expected_starting_index_of_each_representation_h.push_back(4);
    representation_index_mask_h.push_back(3);
    expected_starting_index_of_each_representation_h.push_back(5);
    representation_index_mask_h.push_back(3);
    representation_index_mask_h.push_back(3);
    representation_index_mask_h.push_back(3);
    representation_index_mask_h.push_back(4);
    expected_starting_index_of_each_representation_h.push_back(9);
    representation_index_mask_h.push_back(4);
    representation_index_mask_h.push_back(4);
    representation_index_mask_h.push_back(5);
    expected_starting_index_of_each_representation_h.push_back(12);
    representation_index_mask_h.push_back(6);
    expected_starting_index_of_each_representation_h.push_back(13);

    std::uint32_t number_of_threads = 3;

    test_copy_index_of_first_occurence(representation_index_mask_h,
                                       expected_starting_index_of_each_representation_h,
                                       number_of_threads);
}

TEST(TestCudamapperMatcherGPU, test_copy_index_of_first_occurence_large_example)
{
    const std::uint64_t total_sketch_elements                    = 10000000;
    const std::uint32_t sketch_elements_with_same_representation = 1000;

    thrust::host_vector<std::uint64_t> representation_index_mask_h;
    thrust::host_vector<std::size_t> expected_starting_index_of_each_representation_h;
    for (std::size_t i = 0; i < total_sketch_elements; ++i)
    {
        representation_index_mask_h.push_back(i / sketch_elements_with_same_representation + 1);
        if (i % sketch_elements_with_same_representation == 0)
            expected_starting_index_of_each_representation_h.push_back(i);
    }

    std::uint32_t number_of_threads = 256;

    test_copy_index_of_first_occurence(representation_index_mask_h,
                                       expected_starting_index_of_each_representation_h,
                                       number_of_threads);
}

void test_find_first_occurrences_of_representations(const thrust::host_vector<representation_t>& representations_h,
                                                    const thrust::host_vector<std::uint32_t>& expected_starting_index_of_each_representation_h)
{
    const thrust::device_vector<representation_t> representations_d(representations_h);

    const thrust::device_vector<std::uint32_t> starting_index_of_each_representation_d = details::matcher_gpu::find_first_occurrences_of_representations(representations_d);

    const thrust::host_vector<std::uint32_t> starting_index_of_each_representation_h(starting_index_of_each_representation_d);

    ASSERT_EQ(starting_index_of_each_representation_h.size(), expected_starting_index_of_each_representation_h.size());

    for (std::size_t i = 0; i < expected_starting_index_of_each_representation_h.size(); ++i)
    {
        EXPECT_EQ(starting_index_of_each_representation_h[i], expected_starting_index_of_each_representation_h[i]) << "index: " << i;
    }
}

TEST(TestCudamapperMatcherGPU, test_find_first_occurrences_of_representations_small_example)
{
    /// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    /// 0  0  0  0 12 12 12 12 12 12 23 23 23 32 32 32 32 32 46 46 46
    /// 1  0  0  0  1  0  0  0  0  0  1  0  0  1  0  0  0  0  1  0  0
    /// 1  1  1  1  2  2  2  2  2  2  3  3  3  4  4  4  4  4  5  5  5
    /// ^           ^                 ^        ^              ^       ^
    /// 0  4 10 13 18 21

    thrust::host_vector<representation_t> representations_h;
    thrust::device_vector<std::uint32_t> expected_starting_index_of_each_representation_h;
    representations_h.push_back(0);
    expected_starting_index_of_each_representation_h.push_back(0);
    representations_h.push_back(0);
    representations_h.push_back(0);
    representations_h.push_back(0);
    representations_h.push_back(12);
    expected_starting_index_of_each_representation_h.push_back(4);
    representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(23);
    expected_starting_index_of_each_representation_h.push_back(10);
    representations_h.push_back(23);
    representations_h.push_back(23);
    representations_h.push_back(32);
    expected_starting_index_of_each_representation_h.push_back(13);
    representations_h.push_back(32);
    representations_h.push_back(32);
    representations_h.push_back(32);
    representations_h.push_back(32);
    representations_h.push_back(46);
    expected_starting_index_of_each_representation_h.push_back(18);
    representations_h.push_back(46);
    representations_h.push_back(46);
    expected_starting_index_of_each_representation_h.push_back(21);

    test_find_first_occurrences_of_representations(representations_h,
                                                   expected_starting_index_of_each_representation_h);
}

TEST(TestCudamapperMatcherGPU, test_find_first_occurrences_of_representations_large_example)
{
    const std::uint64_t total_sketch_elements                    = 10000000;
    const std::uint32_t sketch_elements_with_same_representation = 1000;

    thrust::host_vector<representation_t> representations_h;
    thrust::device_vector<std::uint32_t> expected_starting_index_of_each_representation_h;

    for (std::size_t i = 0; i < total_sketch_elements; ++i)
    {
        representations_h.push_back(i / sketch_elements_with_same_representation);
        if (i % sketch_elements_with_same_representation == 0)
        {
            expected_starting_index_of_each_representation_h.push_back(i);
        }
    }
    expected_starting_index_of_each_representation_h.push_back(total_sketch_elements);

    test_find_first_occurrences_of_representations(representations_h,
                                                   expected_starting_index_of_each_representation_h);
}

void test_find_query_target_matches(const thrust::host_vector<representation_t>& query_representations_h,
                                    const thrust::host_vector<representation_t>& target_representations_h,
                                    const thrust::host_vector<std::int64_t>& expected_found_target_indices_h)
{
    const thrust::device_vector<representation_t> query_representations_d(query_representations_h);
    const thrust::device_vector<representation_t> target_representations_d(target_representations_h);
    thrust::device_vector<int64_t> found_target_indices_d(query_representations_d.size());

    details::matcher_gpu::find_query_target_matches(found_target_indices_d, query_representations_d, target_representations_d);

    thrust::device_vector<int64_t> found_target_indices_h(found_target_indices_d);

    ASSERT_EQ(found_target_indices_h.size(), expected_found_target_indices_h.size());

    for (int32_t i = 0; i < get_size(found_target_indices_h); ++i)
    {
        EXPECT_EQ(found_target_indices_h[i], expected_found_target_indices_h[i]) << "index: " << i;
    }
}

TEST(TestCudamapperMatcherGPU, test_find_query_target_matches_small_example)
{
    thrust::host_vector<representation_t> query_representations_h;
    query_representations_h.push_back(0);
    query_representations_h.push_back(12);
    query_representations_h.push_back(23);
    query_representations_h.push_back(32);
    query_representations_h.push_back(46);
    thrust::host_vector<representation_t> target_representations_h;
    target_representations_h.push_back(5);
    target_representations_h.push_back(12);
    target_representations_h.push_back(16);
    target_representations_h.push_back(23);
    target_representations_h.push_back(24);
    target_representations_h.push_back(25);
    target_representations_h.push_back(46);

    thrust::host_vector<int64_t> expected_found_target_indices_h;
    expected_found_target_indices_h.push_back(-1);
    expected_found_target_indices_h.push_back(1);
    expected_found_target_indices_h.push_back(3);
    expected_found_target_indices_h.push_back(-1);
    expected_found_target_indices_h.push_back(6);

    test_find_query_target_matches(query_representations_h, target_representations_h, expected_found_target_indices_h);
}

TEST(TestCudamapperMatcherGPU, test_query_target_matches_large_example)
{
    const std::int64_t total_query_representations = 1000000;

    thrust::host_vector<representation_t> query_representations_h;
    thrust::host_vector<representation_t> target_representations_h;

    for (std::int64_t i = 0; i < total_query_representations; ++i)
    {
        query_representations_h.push_back(i * 3);
    }

    thrust::device_vector<std::int64_t> expected_found_target_indices_h(query_representations_h.size(), -1);

    const representation_t max_representation = query_representations_h.back();
    for (representation_t r = 0; r < max_representation; r += 2)
    {
        target_representations_h.push_back(r);
        if (r % 3 == 0)
        {
            if (r / 3 < expected_found_target_indices_h.size())
            {
                expected_found_target_indices_h[r / 3] = get_size(target_representations_h) - 1;
            }
        }
    }

    test_find_query_target_matches(query_representations_h, target_representations_h, expected_found_target_indices_h);
}

} // namespace cudamapper
} // namespace claragenomics
