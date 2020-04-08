/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <vector>

#include "index_cache.cuh"

namespace claragenomics
{
namespace cudamapper
{

// Functions in this file group Indices in batches of given size
// These batches can then be used to control host and device Index caches

namespace details
{
namespace index_batcher
{

using index_id_t          = std::size_t;
using number_of_indices_t = index_id_t;

/// GroupOfIndicesDescriptor - describes a group of indices by the id of the first index and the total number of indices in that group
struct GroupOfIndicesDescriptor
{
    index_id_t first_index;
    number_of_indices_t number_of_indices;
};

/// GroupAndSubgroupsOfIndicesDescriptor - describes a group of indices and further divides it into subgroups
struct GroupAndSubgroupsOfIndicesDescriptor
{
    GroupOfIndicesDescriptor whole_group;
    std::vector<GroupOfIndicesDescriptor> subgroups;
};

// HostAndDeviceGroupsOfIndices - holds all indices of host batch and device batches
struct HostAndDeviceGroupsOfIndices
{
    std::vector<IndexDescriptor> host_indices_group;
    std::vector<std::vector<IndexDescriptor>> device_indices_groups;
};

/// \brief Splits numbers into groups
///
/// Splits numbers between first_index and first_index + number_of_indices into groups of indices_per_group numbers.
/// If number_of_indices is not divisible by indices_per_group last group will have less elements.
///
/// For example for:
/// first_index = 16
/// number_of_indices = 25
/// indices_per_group = 7
/// generated groups will be:
/// (16, 7), (23, 7), (30, 7), (37, 4)
///
/// \param first_index
/// \param number_of_indices
/// \param indices_per_group
/// \return generated groups
std::vector<GroupOfIndicesDescriptor> split_array_into_groups(const index_id_t first_index,
                                                              const number_of_indices_t number_of_indices,
                                                              const number_of_indices_t indices_per_group);

/// \brief Splits numbers into groups and then further splits each group into subgroups
///
/// Splits numbers between first_index and first_index + number_of_indices into groups of indices_per_group numbers.
/// If number_of_indices is not divisible by indices_per_group last group will have less elements.
/// After this it splits each group into subgroups of indices_per_subgroup elements.
///
/// For example for:
/// first_index = 100
/// number_of_indices = 71
/// indices_per_group = 16
/// indices_per_subgroup = 5
/// generated groups will be:
/// (100, 16) - (100, 5), (105, 5), (110, 5), (115, 1)
/// (116, 16) - (116, 5), (121, 5), (126, 5), (131, 1)
/// (132, 16) - (132, 5), (137, 5), (142, 5), (147, 1)
/// (148, 16) - (148, 5), (153, 5), (158, 5), (163, 1)
/// (164,  7) - (164, 5), (169, 2)
///
/// \param first_index
/// \param total_number_of_indices
/// \param indices_per_group
/// \param indices_per_subgroup
/// \return generated groups
std::vector<GroupAndSubgroupsOfIndicesDescriptor> generate_groups_and_subgroups(const index_id_t first_index,
                                                                                const number_of_indices_t total_number_of_indices,
                                                                                const number_of_indices_t indices_per_group,
                                                                                const number_of_indices_t indices_per_subgroup);

/// \brief Transforms descriptor of group of indices into descriptors of indices
/// \param index_descriptors descriptor of every individual index
/// \param groups_and_subgroups descriptors of groups of indices
/// \return groups of index descriptors
std::vector<HostAndDeviceGroupsOfIndices> convert_groups_of_indices_into_groups_of_index_descriptors(const std::vector<IndexDescriptor>& index_descriptors,
                                                                                                     const std::vector<GroupAndSubgroupsOfIndicesDescriptor>& groups_and_subgroups);

} // namespace index_batcher
} // namespace details

} // namespace cudamapper
} // namespace claragenomics
