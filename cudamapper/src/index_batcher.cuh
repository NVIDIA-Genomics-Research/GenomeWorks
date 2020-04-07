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

/// \brief Splits numbers into groups
///
/// Splits numbers between first_index and first_index + number_of_indies into groups of indices_per_group numbers.
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

} // namespace index_batcher
} // namespace details

} // namespace cudamapper
} // namespace claragenomics