/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "index_cache.cuh"

namespace claragenomics
{
namespace cudamapper
{

bool operator==(const IndexDescriptor& lhs, const IndexDescriptor& rhs)
{
    return lhs.first_read == rhs.first_read && lhs.number_of_reads == rhs.number_of_reads;
}

bool operator!=(const IndexDescriptor& lhs, const IndexDescriptor& rhs)
{
    return !(lhs == rhs);
}

std::size_t IndexDescriptorHash::operator()(const IndexDescriptor& index_descriptor) const
{
    std::size_t hash = 0;

    // populate lower half of hash with one value, upper half with the other
    std::size_t bytes_per_element = sizeof(std::size_t) / 2;
    // first half of element_mask are zeros, second are ones
    std::size_t element_mask = (1 << (8 * bytes_per_element)) - 1;
    // set lower bits
    hash |= index_descriptor.first_read & element_mask;
    // set higher bits
    hash |= (index_descriptor.number_of_reads & element_mask) << (8 * bytes_per_element);

    return hash;
}

} // namespace cudamapper
} // namespace claragenomics