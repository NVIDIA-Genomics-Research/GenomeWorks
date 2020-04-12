/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "index_batcher.cuh"

#include <algorithm>

namespace claragenomics
{
namespace cudamapper
{

namespace details
{
namespace index_batcher
{

std::vector<GroupOfIndicesDescriptor> split_array_into_groups(const index_id_t first_index,
                                                              const number_of_indices_t number_of_indices,
                                                              const number_of_indices_t indices_per_group)
{
    std::vector<GroupOfIndicesDescriptor> groups;
    const index_id_t past_the_last_index = first_index + number_of_indices;

    for (index_id_t first_index_in_group = first_index; first_index_in_group < past_the_last_index; first_index_in_group += indices_per_group)
    {
        const number_of_indices_t indices_in_this_group = std::min(indices_per_group, past_the_last_index - first_index_in_group);
        groups.push_back({first_index_in_group, indices_in_this_group});
    }

    return groups;
}

std::vector<GroupAndSubgroupsOfIndicesDescriptor> generate_groups_and_subgroups(const index_id_t first_index,
                                                                                const number_of_indices_t total_number_of_indices,
                                                                                const number_of_indices_t indices_per_group,
                                                                                const number_of_indices_t indices_per_subgroup)
{
    std::vector<GroupAndSubgroupsOfIndicesDescriptor> groups_and_subroups;

    std::vector<GroupOfIndicesDescriptor> main_groups = split_array_into_groups(first_index,
                                                                                total_number_of_indices,
                                                                                indices_per_group);

    for (const GroupOfIndicesDescriptor& main_group : main_groups)
    {
        std::vector<GroupOfIndicesDescriptor> subgroups = split_array_into_groups(main_group.first_index,
                                                                                  main_group.number_of_indices,
                                                                                  indices_per_subgroup);

        groups_and_subroups.push_back({main_group, subgroups});
    }

    return groups_and_subroups;
}

std::vector<HostAndDeviceGroupsOfIndices> convert_groups_of_indices_into_groups_of_index_descriptors(const std::vector<IndexDescriptor>& index_descriptors,
                                                                                                     const std::vector<GroupAndSubgroupsOfIndicesDescriptor>& groups_and_subgroups)
{
    std::vector<HostAndDeviceGroupsOfIndices> results;

    for (const details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor& group_and_its_subgroups : groups_and_subgroups)
    {
        details::index_batcher::GroupOfIndicesDescriptor host_group = group_and_its_subgroups.whole_group;
        std::vector<IndexDescriptor> host_indices;
        std::copy(std::begin(index_descriptors) + host_group.first_index,
                  std::begin(index_descriptors) + host_group.first_index + host_group.number_of_indices,
                  std::back_inserter(host_indices));

        std::vector<details::index_batcher::GroupOfIndicesDescriptor> device_groups = group_and_its_subgroups.subgroups;
        std::vector<std::vector<IndexDescriptor>> device_indices;
        for (const details::index_batcher::GroupOfIndicesDescriptor& device_group : device_groups)
        {
            std::vector<IndexDescriptor> device_indices_single;
            std::copy(std::begin(index_descriptors) + device_group.first_index,
                      std::begin(index_descriptors) + device_group.first_index + device_group.number_of_indices,
                      std::back_inserter(device_indices_single));
            device_indices.push_back(std::move(device_indices_single));
        }
        results.push_back({host_indices, device_indices});
    }

    return results;
}

std::vector<BatchOfIndices> combine_query_and_target_indices(const std::vector<HostAndDeviceGroupsOfIndices>& query_groups_of_indices,
                                                             const std::vector<HostAndDeviceGroupsOfIndices>& target_groups_of_indices)
{
    std::vector<BatchOfIndices> all_batches;

    for (const HostAndDeviceGroupsOfIndices& query_group_of_indices : query_groups_of_indices)
    {
        const std::vector<IndexDescriptor>& query_host_group_of_indices              = query_group_of_indices.host_indices_group;
        const std::vector<std::vector<IndexDescriptor>>& query_device_indices_groups = query_group_of_indices.device_indices_groups;
        for (const HostAndDeviceGroupsOfIndices& target_group_of_indices : target_groups_of_indices)
        {
            const std::vector<IndexDescriptor>& target_host_group_of_indices = target_group_of_indices.host_indices_group;
            IndexBatch host_batch{query_host_group_of_indices, target_host_group_of_indices};

            const std::vector<std::vector<IndexDescriptor>>& target_device_indices_groups = target_group_of_indices.device_indices_groups;
            std::vector<IndexBatch> device_batches;

            for (const std::vector<IndexDescriptor>& query_device_indices_group : query_device_indices_groups)
            {
                for (const std::vector<IndexDescriptor>& target_device_indices_group : target_device_indices_groups)
                {
                    device_batches.push_back({query_device_indices_group, target_device_indices_group});
                }
            }

            all_batches.push_back({host_batch, device_batches});
        }
    }

    return all_batches;
}

} // namespace index_batcher
} // namespace details

} // namespace cudamapper
} // namespace claragenomics
