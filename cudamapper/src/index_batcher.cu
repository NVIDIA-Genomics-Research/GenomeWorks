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
#include <exception>

namespace claragenomics
{
namespace cudamapper
{

std::vector<BatchOfIndices> generate_batches_of_indices(const number_of_indices_t query_indices_per_host_batch,
                                                        const number_of_indices_t query_indices_per_device_batch,
                                                        const number_of_indices_t target_indices_per_host_batch,
                                                        const number_of_indices_t target_indices_per_device_batch,
                                                        const std::shared_ptr<const claragenomics::io::FastaParser> query_parser,
                                                        const std::shared_ptr<const claragenomics::io::FastaParser> target_parser,
                                                        const number_of_basepairs_t query_basepairs_per_index,
                                                        const number_of_basepairs_t target_basepairs_per_index,
                                                        const bool same_query_and_target)
{
    if (same_query_and_target)
    {
        if (query_indices_per_host_batch != target_indices_per_host_batch)
        {
            throw std::invalid_argument("generate_batches_of_indices: indices_per_host_batch not the same");
        }
        if (query_indices_per_device_batch != target_indices_per_device_batch)
        {
            throw std::invalid_argument("generate_batches_of_indices: indices_per_device_batch not the same");
        }
        if (query_parser != target_parser)
        {
            throw std::invalid_argument("generate_batches_of_indices: parser not the same");
        }
        if (query_basepairs_per_index != target_basepairs_per_index)
        {
            throw std::invalid_argument("generate_batches_of_indices: basepairs_per_index not the same");
        }
    }

    // get IndexDescriptors for all indices
    // TODO: modify FastaParser so it returns IndexDescriptors
    std::vector<std::pair<int, int>> query_chunks = query_parser->get_read_chunks(query_basepairs_per_index);
    std::vector<IndexDescriptor> query_index_descriptors;
    for (const auto& query_chunk : query_chunks)
    {
        // first_read, number_of_reads
        query_index_descriptors.push_back({static_cast<read_id_t>(query_chunk.first), static_cast<read_id_t>(query_chunk.second - query_chunk.first)});
    }
    std::vector<std::pair<int, int>> target_chunks = target_parser->get_read_chunks(target_basepairs_per_index);
    std::vector<IndexDescriptor> target_index_descriptors;
    for (const auto& target_chunk : target_chunks)
    {
        // first_read, number_of_reads
        target_index_descriptors.push_back({static_cast<read_id_t>(target_chunk.first), static_cast<read_id_t>(target_chunk.second - target_chunk.first)});
    }

    // find out which Indices go in which batches
    const std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor> query_groups_and_subgroups =
        details::index_batcher::generate_groups_and_subgroups(0,
                                                              query_index_descriptors.size(),
                                                              query_indices_per_host_batch,
                                                              query_indices_per_device_batch);

    std::vector<details::index_batcher::HostAndDeviceGroupsOfIndices> query_index_descriptors_groups_and_subgroups =
        convert_groups_of_indices_into_groups_of_index_descriptors(query_index_descriptors,
                                                                   query_groups_and_subgroups);

    const std::vector<details::index_batcher::GroupAndSubgroupsOfIndicesDescriptor> target_groups_and_subgroups =
        details::index_batcher::generate_groups_and_subgroups(0,
                                                              target_index_descriptors.size(),
                                                              target_indices_per_host_batch,
                                                              target_indices_per_device_batch);

    std::vector<details::index_batcher::HostAndDeviceGroupsOfIndices> target_index_descriptors_groups_and_subgroups =
        convert_groups_of_indices_into_groups_of_index_descriptors(target_index_descriptors,
                                                                   target_groups_and_subgroups);

    // combine query and target index groups and return
    return combine_query_and_target_indices(query_index_descriptors_groups_and_subgroups,
                                            target_index_descriptors_groups_and_subgroups,
                                            same_query_and_target);
}

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
                                                             const std::vector<HostAndDeviceGroupsOfIndices>& target_groups_of_indices,
                                                             const bool same_query_and_target)
{
    assert(!same_query_and_target || (query_groups_of_indices == target_groups_of_indices));

    std::vector<BatchOfIndices> all_batches;

    for (std::size_t query_host_id = 0; query_host_id < query_groups_of_indices.size(); ++query_host_id)
    {
        const HostAndDeviceGroupsOfIndices& query_group_of_indices                   = query_groups_of_indices[query_host_id];
        const std::vector<IndexDescriptor>& query_host_group_of_indices              = query_group_of_indices.host_indices_group;
        const std::vector<std::vector<IndexDescriptor>>& query_device_indices_groups = query_group_of_indices.device_indices_groups;
        // if same_query_and_target == true only generated batches where target_host_id >= query_host_id
        // combinations where target_host_id < query_host_id are not needed due to symmetry
        for (std::size_t target_host_id = same_query_and_target ? query_host_id : 0;
             target_host_id < target_groups_of_indices.size();
             ++target_host_id)
        {
            const HostAndDeviceGroupsOfIndices& target_group_of_indices      = target_groups_of_indices[target_host_id];
            const std::vector<IndexDescriptor>& target_host_group_of_indices = target_group_of_indices.host_indices_group;
            IndexBatch host_batch{query_host_group_of_indices, target_host_group_of_indices};

            const std::vector<std::vector<IndexDescriptor>>& target_device_indices_groups = target_group_of_indices.device_indices_groups;
            std::vector<IndexBatch> device_batches;

            for (std::size_t query_device_id = 0; query_device_id < query_device_indices_groups.size(); ++query_device_id)
            {
                const std::vector<IndexDescriptor>& query_device_indices_group = query_device_indices_groups[query_device_id];
                // if same_query_and_target == true and target_host_id == query_host_id only generated batches where target_device_id >= query_device_id
                // combinations where target_host_id < query_host_id are not needed due to symmetry
                for (std::size_t target_device_id = (same_query_and_target && target_host_id == query_host_id) ? query_device_id : 0;
                     target_device_id < target_device_indices_groups.size();
                     ++target_device_id)
                {
                    const std::vector<IndexDescriptor>& target_device_indices_group = target_device_indices_groups[target_device_id];
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
