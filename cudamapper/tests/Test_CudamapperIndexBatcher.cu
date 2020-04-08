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

// *** test split_array_into_groups ***

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

// *** test generate_groups_and_subgroups ***

void test_generate_groups_and_subgroups(const details::index_batcher::index_id_t first_index,
                                        const details::index_batcher::number_of_indices_t total_number_of_indices,
                                        const details::index_batcher::number_of_indices_t indices_per_group,
                                        const details::index_batcher::number_of_indices_t indices_per_subgroup,
                                        const std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor>& expected_groups)
{
    std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor> generated_groups = details::index_batcher::generate_groups_and_subgroups(first_index,
                                                                                                                                                       total_number_of_indices,
                                                                                                                                                       indices_per_group,
                                                                                                                                                       indices_per_subgroup);

    ASSERT_EQ(expected_groups.size(), generated_groups.size());
    for (std::size_t i = 0; i < expected_groups.size(); ++i)
    {
        const auto& expected_main_group  = expected_groups[i].whole_group;
        const auto& generated_main_group = generated_groups[i].whole_group;
        ASSERT_EQ(expected_main_group.first_index, generated_main_group.first_index) << "i: " << i;
        ASSERT_EQ(expected_main_group.number_of_indices, generated_main_group.number_of_indices) << "i: " << i;
        const auto& expected_subgroups  = expected_groups[i].subgroups;
        const auto& generated_subgroups = generated_groups[i].subgroups;
        ASSERT_EQ(expected_subgroups.size(), generated_subgroups.size()) << "i: " << i;
        for (std::size_t j = 0; j < expected_subgroups.size(); ++j)
        {
            const auto& expected_subgroup  = expected_subgroups[j];
            const auto& generated_subgroup = generated_subgroups[j];
            ASSERT_EQ(expected_subgroup.first_index, generated_subgroup.first_index) << "i: " << i << ", j: " << j;
            ASSERT_EQ(expected_subgroup.number_of_indices, generated_subgroup.number_of_indices) << "i: " << i << ", j: " << j;
        }
    }
}

TEST(TestCudamapperIndexBatcher, test_split_array_into_groups_all_divisible)
{
    const details::index_batcher::index_id_t first_index                      = 528;
    const details::index_batcher::number_of_indices_t total_number_of_indices = 64;
    const details::index_batcher::number_of_indices_t indices_per_group       = 8;
    const details::index_batcher::number_of_indices_t indices_per_subgroup    = 4;

    std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor> expected_groups;
    expected_groups.push_back({{528, 8}, {}});
    expected_groups.back().subgroups.push_back({528, 4});
    expected_groups.back().subgroups.push_back({532, 4});
    expected_groups.push_back({{536, 8}, {}});
    expected_groups.back().subgroups.push_back({536, 4});
    expected_groups.back().subgroups.push_back({540, 4});
    expected_groups.push_back({{544, 8}, {}});
    expected_groups.back().subgroups.push_back({544, 4});
    expected_groups.back().subgroups.push_back({548, 4});
    expected_groups.push_back({{552, 8}, {}});
    expected_groups.back().subgroups.push_back({552, 4});
    expected_groups.back().subgroups.push_back({556, 4});
    expected_groups.push_back({{560, 8}, {}});
    expected_groups.back().subgroups.push_back({560, 4});
    expected_groups.back().subgroups.push_back({564, 4});
    expected_groups.push_back({{568, 8}, {}});
    expected_groups.back().subgroups.push_back({568, 4});
    expected_groups.back().subgroups.push_back({572, 4});
    expected_groups.push_back({{576, 8}, {}});
    expected_groups.back().subgroups.push_back({576, 4});
    expected_groups.back().subgroups.push_back({580, 4});
    expected_groups.push_back({{584, 8}, {}});
    expected_groups.back().subgroups.push_back({584, 4});
    expected_groups.back().subgroups.push_back({588, 4});

    test_generate_groups_and_subgroups(first_index,
                                       total_number_of_indices,
                                       indices_per_group,
                                       indices_per_subgroup,
                                       expected_groups);
}

TEST(TestCudamapperIndexBatcher, test_split_array_into_groups_subgroup_not_divisible)
{
    const details::index_batcher::index_id_t first_index                      = 528;
    const details::index_batcher::number_of_indices_t total_number_of_indices = 64;
    const details::index_batcher::number_of_indices_t indices_per_group       = 8;
    const details::index_batcher::number_of_indices_t indices_per_subgroup    = 5;

    std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor> expected_groups;
    expected_groups.push_back({{528, 8}, {}});
    expected_groups.back().subgroups.push_back({528, 5});
    expected_groups.back().subgroups.push_back({533, 3});
    expected_groups.push_back({{536, 8}, {}});
    expected_groups.back().subgroups.push_back({536, 5});
    expected_groups.back().subgroups.push_back({541, 3});
    expected_groups.push_back({{544, 8}, {}});
    expected_groups.back().subgroups.push_back({544, 5});
    expected_groups.back().subgroups.push_back({549, 3});
    expected_groups.push_back({{552, 8}, {}});
    expected_groups.back().subgroups.push_back({552, 5});
    expected_groups.back().subgroups.push_back({557, 3});
    expected_groups.push_back({{560, 8}, {}});
    expected_groups.back().subgroups.push_back({560, 5});
    expected_groups.back().subgroups.push_back({565, 3});
    expected_groups.push_back({{568, 8}, {}});
    expected_groups.back().subgroups.push_back({568, 5});
    expected_groups.back().subgroups.push_back({573, 3});
    expected_groups.push_back({{576, 8}, {}});
    expected_groups.back().subgroups.push_back({576, 5});
    expected_groups.back().subgroups.push_back({581, 3});
    expected_groups.push_back({{584, 8}, {}});
    expected_groups.back().subgroups.push_back({584, 5});
    expected_groups.back().subgroups.push_back({589, 3});

    test_generate_groups_and_subgroups(first_index,
                                       total_number_of_indices,
                                       indices_per_group,
                                       indices_per_subgroup,
                                       expected_groups);
}

TEST(TestCudamapperIndexBatcher, test_split_array_into_groups_nothing_divisible)
{
    const details::index_batcher::index_id_t first_index                      = 528;
    const details::index_batcher::number_of_indices_t total_number_of_indices = 66;
    const details::index_batcher::number_of_indices_t indices_per_group       = 8;
    const details::index_batcher::number_of_indices_t indices_per_subgroup    = 5;

    std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor> expected_groups;
    expected_groups.push_back({{528, 8}, {}});
    expected_groups.back().subgroups.push_back({528, 5});
    expected_groups.back().subgroups.push_back({533, 3});
    expected_groups.push_back({{536, 8}, {}});
    expected_groups.back().subgroups.push_back({536, 5});
    expected_groups.back().subgroups.push_back({541, 3});
    expected_groups.push_back({{544, 8}, {}});
    expected_groups.back().subgroups.push_back({544, 5});
    expected_groups.back().subgroups.push_back({549, 3});
    expected_groups.push_back({{552, 8}, {}});
    expected_groups.back().subgroups.push_back({552, 5});
    expected_groups.back().subgroups.push_back({557, 3});
    expected_groups.push_back({{560, 8}, {}});
    expected_groups.back().subgroups.push_back({560, 5});
    expected_groups.back().subgroups.push_back({565, 3});
    expected_groups.push_back({{568, 8}, {}});
    expected_groups.back().subgroups.push_back({568, 5});
    expected_groups.back().subgroups.push_back({573, 3});
    expected_groups.push_back({{576, 8}, {}});
    expected_groups.back().subgroups.push_back({576, 5});
    expected_groups.back().subgroups.push_back({581, 3});
    expected_groups.push_back({{584, 8}, {}});
    expected_groups.back().subgroups.push_back({584, 5});
    expected_groups.back().subgroups.push_back({589, 3});
    expected_groups.push_back({{592, 2}, {}});
    expected_groups.back().subgroups.push_back({592, 2});

    test_generate_groups_and_subgroups(first_index,
                                       total_number_of_indices,
                                       indices_per_group,
                                       indices_per_subgroup,
                                       expected_groups);
}

// *** test convert_groups_of_indices_into_groups_of_index_descriptors ***

void test_convert_groups_of_indices_into_groups_of_index_descriptors(const std::vector<IndexDescriptor>& index_descriptors,
                                                                     const std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor>& groups_and_subgroups,
                                                                     const std::vector<details::index_batcher::HostAndDeviceGroupsOfIndices>& expected_groups_of_indices)
{
    const std::vector<details::index_batcher::HostAndDeviceGroupsOfIndices> generated_groups_of_indices =
        details::index_batcher::convert_groups_of_indices_into_groups_of_index_descriptors(index_descriptors,
                                                                                           groups_and_subgroups);

    ASSERT_EQ(expected_groups_of_indices.size(), generated_groups_of_indices.size());

    for (std::size_t i = 0; i < expected_groups_of_indices.size(); ++i)
    {
        const details::index_batcher::HostAndDeviceGroupsOfIndices& expected_group_of_indices  = expected_groups_of_indices[i];
        const details::index_batcher::HostAndDeviceGroupsOfIndices& generated_group_of_indices = generated_groups_of_indices[i];
        // check indices in host batch
        ASSERT_EQ(expected_group_of_indices.host_indices_group.size(), generated_group_of_indices.host_indices_group.size()) << "i: " << i;
        for (std::size_t j = 0; j < expected_group_of_indices.host_indices_group.size(); ++j)
        {
            const IndexDescriptor& expected_host_index  = expected_group_of_indices.host_indices_group[j];
            const IndexDescriptor& generated_host_index = generated_group_of_indices.host_indices_group[j];
            ASSERT_EQ(expected_host_index.first_read(), generated_host_index.first_read()) << "i: " << i << ", j: " << j;
            ASSERT_EQ(expected_host_index.number_of_reads(), generated_host_index.number_of_reads()) << "i: " << i << ", j: " << j;
        }
        // check indices in multiple device batches
        ASSERT_EQ(expected_group_of_indices.device_indices_groups.size(), generated_group_of_indices.device_indices_groups.size()) << "i: " << i;
        for (std::size_t j = 0; j < expected_group_of_indices.device_indices_groups.size(); ++j)
        {
            const std::vector<IndexDescriptor>& expected_device_indices_group  = expected_group_of_indices.device_indices_groups[j];
            const std::vector<IndexDescriptor>& generated_device_indices_group = generated_group_of_indices.device_indices_groups[j];
            // check indices in every device batch
            ASSERT_EQ(expected_device_indices_group.size(), generated_device_indices_group.size());
            for (std::size_t k = 0; k < expected_device_indices_group.size(); ++k)
            {
                const IndexDescriptor& expected_host_index  = expected_device_indices_group[k];
                const IndexDescriptor& generated_host_index = generated_device_indices_group[k];
                ASSERT_EQ(expected_host_index.first_read(), generated_host_index.first_read()) << "i: " << i << ", j: " << j << "k: " << k;
                ASSERT_EQ(expected_host_index.number_of_reads(), generated_host_index.number_of_reads()) << "i: " << i << ", j: " << j << "k: " << k;
            }
        }
    }
}

TEST(TestCudamapperIndexBatcher, test_convert_groups_of_indices_into_groups_of_index_descriptors)
{
    std::vector<IndexDescriptor> index_descriptors;
    index_descriptors.push_back({0, 10});   // 0
    index_descriptors.push_back({10, 10});  // 1
    index_descriptors.push_back({20, 10});  // 2
    index_descriptors.push_back({30, 10});  // 3
    index_descriptors.push_back({40, 10});  // 4
    index_descriptors.push_back({50, 20});  // 5
    index_descriptors.push_back({70, 20});  // 6
    index_descriptors.push_back({90, 30});  // 7
    index_descriptors.push_back({120, 50}); // 8
    index_descriptors.push_back({170, 50}); // 9

    std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor> groups_and_subgroups;
    std::vector<details::index_batcher::HostAndDeviceGroupsOfIndices> expected_groups_of_indices;
    groups_and_subgroups.push_back({{0, 3}, {{0, 1}, {1, 2}}});
    expected_groups_of_indices.push_back({{{0, 10}, {10, 10}, {20, 10}},
                                          {{{0, 10}}, {{10, 10}, {20, 10}}}});
    groups_and_subgroups.push_back({{3, 4}, {{3, 2}, {5, 2}}});
    expected_groups_of_indices.push_back({{{30, 10}, {40, 10}, {50, 20}, {70, 20}},
                                          {{{30, 10}, {40, 10}}, {{50, 20}, {70, 20}}}});
    groups_and_subgroups.push_back({{7, 3}, {{7, 3}}});
    expected_groups_of_indices.push_back({{{90, 30}, {120, 50}, {170, 50}},
                                          {{{90, 30}, {120, 50}, {170, 50}}}});

    test_convert_groups_of_indices_into_groups_of_index_descriptors(index_descriptors,
                                                                    groups_and_subgroups,
                                                                    expected_groups_of_indices);
}

} // namespace cudamapper
} // namespace claragenomics
