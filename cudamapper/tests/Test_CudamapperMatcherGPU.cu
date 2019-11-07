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

void test_compute_number_of_anchors(const thrust::host_vector<std::uint32_t>& query_starting_index_of_each_representation_h,
                                    const thrust::host_vector<std::int64_t>& found_target_indices_h,
                                    const thrust::host_vector<std::uint32_t>& target_starting_index_of_each_representation_h,
                                    const thrust::host_vector<std::int64_t>& expected_anchor_starting_indices_h)
{
    const thrust::device_vector<std::uint32_t> query_starting_index_of_each_representation_d(query_starting_index_of_each_representation_h);
    const thrust::device_vector<std::uint32_t> target_starting_index_of_each_representation_d(target_starting_index_of_each_representation_h);
    const thrust::device_vector<std::int64_t> found_target_indices_d(found_target_indices_h);
    thrust::device_vector<std::int64_t> anchor_starting_indices_d(found_target_indices_h.size());

    details::matcher_gpu::compute_anchor_starting_indices(anchor_starting_indices_d, query_starting_index_of_each_representation_d, found_target_indices_d, target_starting_index_of_each_representation_d);

    thrust::host_vector<std::int64_t> anchor_starting_indices_h(anchor_starting_indices_d);

    for (int32_t i = 0; i < get_size(found_target_indices_h); ++i)
    {
        EXPECT_EQ(anchor_starting_indices_h[i], expected_anchor_starting_indices_h[i]);
    }
}

TEST(TestCudamapperMatcherGPU, test_compute_number_of_anchors_small_example)
{
    thrust::host_vector<representation_t> query_starting_index_of_each_representation_h;
    query_starting_index_of_each_representation_h.push_back(0);
    query_starting_index_of_each_representation_h.push_back(4);
    query_starting_index_of_each_representation_h.push_back(10);
    query_starting_index_of_each_representation_h.push_back(13);
    query_starting_index_of_each_representation_h.push_back(18);
    query_starting_index_of_each_representation_h.push_back(21);

    thrust::host_vector<representation_t> target_starting_index_of_each_representation_h;
    target_starting_index_of_each_representation_h.push_back(0);
    target_starting_index_of_each_representation_h.push_back(3);
    target_starting_index_of_each_representation_h.push_back(7);
    target_starting_index_of_each_representation_h.push_back(9);
    target_starting_index_of_each_representation_h.push_back(13);
    target_starting_index_of_each_representation_h.push_back(16);
    target_starting_index_of_each_representation_h.push_back(18);
    target_starting_index_of_each_representation_h.push_back(21);

    thrust::host_vector<int64_t> found_target_indices_h;
    found_target_indices_h.push_back(-1);
    found_target_indices_h.push_back(1);
    found_target_indices_h.push_back(3);
    found_target_indices_h.push_back(-1);
    found_target_indices_h.push_back(6);

    thrust::host_vector<int64_t> expected_anchor_starting_indices;
    expected_anchor_starting_indices.push_back(0);
    expected_anchor_starting_indices.push_back(24);
    expected_anchor_starting_indices.push_back(36);
    expected_anchor_starting_indices.push_back(36);
    expected_anchor_starting_indices.push_back(45);

    test_compute_number_of_anchors(query_starting_index_of_each_representation_h,
                                   found_target_indices_h,
                                   target_starting_index_of_each_representation_h,
                                   expected_anchor_starting_indices);
}

TEST(TestCudamapperMatcherGPU, test_compute_number_of_anchors_large_example)
{
    const std::int64_t length = 100000;

    thrust::host_vector<representation_t> query_starting_index_of_each_representation_h;
    thrust::host_vector<representation_t> target_starting_index_of_each_representation_h;
    thrust::host_vector<std::int64_t> found_target_indices_h(length - 1, -1);
    thrust::host_vector<std::int64_t> expected_anchor_starting_indices_h;
    std::int64_t expected_n_anchors = 0;
    for (std::int64_t i = 0; i < length; ++i)
    {
        query_starting_index_of_each_representation_h.push_back(2 * i);
        target_starting_index_of_each_representation_h.push_back(10 * i + i % 10);
        if (i % 3 == 0 && i < length - 1)
        {
            found_target_indices_h[i] = i;
            expected_n_anchors += 2 * (10 + (i + 1) % 10 - i % 10);
        }
        if (i < length - 1)
            expected_anchor_starting_indices_h.push_back(expected_n_anchors);
    }

    test_compute_number_of_anchors(query_starting_index_of_each_representation_h,
                                   found_target_indices_h,
                                   target_starting_index_of_each_representation_h,
                                   expected_anchor_starting_indices_h);
}

} // namespace cudamapper
} // namespace claragenomics
