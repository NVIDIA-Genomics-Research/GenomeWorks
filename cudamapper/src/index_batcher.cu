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

} // namespace index_batcher
} // namespace details

} // namespace cudamapper
} // namespace claragenomics