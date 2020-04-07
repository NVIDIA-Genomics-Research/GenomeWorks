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

} // namespace index_batcher
} // namespace details

} // namespace cudamapper
} // namespace claragenomics
