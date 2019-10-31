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

#include "../src/matcher_gpu.cuh"

namespace claragenomics
{

namespace cudamapper
{

void test_create_new_value_mask(const thrust::host_vector<representation_t>& representations_h,
                                const thrust::host_vector<std::uint8_t>& expected_new_value_mask_h,
                                std::uint32_t number_of_threads)
{
    thrust::device_vector<representation_t> representations_d(representations_h);
    thrust::device_vector<std::uint8_t> new_value_mask_d(representations_h.size());

    std::uint32_t number_of_blocks = (representations_h.size() - 1) / number_of_threads + 1;

    details::matcher_gpu::create_new_value_mask<<<number_of_blocks, number_of_threads>>>(thrust::raw_pointer_cast(representations_d.data()),
                                                                                         representations_d.size(),
                                                                                         thrust::raw_pointer_cast(new_value_mask_d.data()));

    CGA_CU_CHECK_ERR(cudaDeviceSynchronize());

    thrust::host_vector<std::uint8_t> new_value_mask_h(new_value_mask_d);

    ASSERT_EQ(new_value_mask_h.size(), expected_new_value_mask_h.size());
    for (std::size_t i = 0; i < expected_new_value_mask_h.size(); ++i)
    {
        EXPECT_EQ(new_value_mask_h[i], expected_new_value_mask_h[i]) << "index: " << i;
    }
}

TEST(TestCudamapperMatcherGPU, test_create_new_value_mask_small_example)
{
    thrust::host_vector<representation_t> representations_h;
    thrust::host_vector<std::uint8_t> expected_new_value_mask_h;
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
    std::uint64_t total_sketch_elements                    = 10000000;
    std::uint32_t sketch_elements_with_same_representation = 1000;

    thrust::host_vector<representation_t> representations_h;
    thrust::host_vector<std::uint8_t> expected_new_value_mask_h;
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

} // namespace cudamapper
} // namespace claragenomics
