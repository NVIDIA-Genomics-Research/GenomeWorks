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

#include "../src/index_batcher.cuh"

namespace claragenomics
{
namespace cudamapper
{

void test_split_array_into_groups(const details::index_batcher::index_id_t first_index,
                                  const details::index_batcher::number_of_indices_t number_of_indices,
                                  const details::index_batcher::number_of_indices_t indices_per_group,
                                  const std::vector<details::index_batcher::GroupOfIndicesDescriptor>& expected_groups)
{
    std::vector<details::index_batcher::GroupOfIndicesDescriptor> generated_groups = details::index_batcher::split_array_into_groups(first_index,
                                                                                                                                     number_of_indices,
                                                                                                                                     indices_per_group);

    ASSERT_EQ(expected_groups.size(), generated_groups.size());
    for (std::size_t i = 0; i < expected_groups.size(); ++i)
    {
        ASSERT_EQ(expected_groups[i].first_index, generated_groups[i].first_index) << "i: " << i;
        ASSERT_EQ(expected_groups[i].number_of_indices, generated_groups[i].number_of_indices) << "i: " << i;
    }
}

TEST(TestCudamapperIndexBatcher, test_split_array_into_groups_divisible)
{
    const details::index_batcher::index_id_t first_index                = 7;
    const details::index_batcher::number_of_indices_t number_of_indices = 16;
    const details::index_batcher::number_of_indices_t indices_per_group = 4;

    std::vector<details::index_batcher::GroupOfIndicesDescriptor> expected_groups;
    expected_groups.push_back({7, 4});
    expected_groups.push_back({11, 4});
    expected_groups.push_back({15, 4});
    expected_groups.push_back({19, 4});

    test_split_array_into_groups(first_index,
                                 number_of_indices,
                                 indices_per_group,
                                 expected_groups);
}

TEST(TestCudamapperIndexBatcher, test_split_array_into_groups_not_divisible)
{
    const details::index_batcher::index_id_t first_index                = 13;
    const details::index_batcher::number_of_indices_t number_of_indices = 58;
    const details::index_batcher::number_of_indices_t indices_per_group = 5;

    std::vector<details::index_batcher::GroupOfIndicesDescriptor> expected_groups;
    expected_groups.push_back({13, 5});
    expected_groups.push_back({18, 5});
    expected_groups.push_back({23, 5});
    expected_groups.push_back({28, 5});
    expected_groups.push_back({33, 5});
    expected_groups.push_back({38, 5});
    expected_groups.push_back({43, 5});
    expected_groups.push_back({48, 5});
    expected_groups.push_back({53, 5});
    expected_groups.push_back({58, 5});
    expected_groups.push_back({63, 5});
    expected_groups.push_back({68, 3});

    test_split_array_into_groups(first_index,
                                 number_of_indices,
                                 indices_per_group,
                                 expected_groups);
}

} // namespace cudamapper
} // namespace claragenomics