/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "index_descriptor.hpp"

#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claragenomics
{
namespace cudamapper
{

IndexDescriptor::IndexDescriptor(read_id_t first_read,
                                 number_of_reads_t number_of_reads)
    : first_read_(first_read)
    , number_of_reads_(number_of_reads)
    , hash_(0)
{
    generate_hash();
}

read_id_t IndexDescriptor::first_read() const
{
    return first_read_;
}

number_of_reads_t IndexDescriptor::number_of_reads() const
{
    return number_of_reads_;
}

std::size_t IndexDescriptor::get_hash() const
{
    return hash_;
}

void IndexDescriptor::generate_hash()
{
    static_assert(sizeof(std::size_t) == 8, "only 64-bit values supported, adjust element_mask and shift_bits");

    // populate lower half of hash with one value, upper half with the other
    constexpr std::size_t element_mask = 0xFFFFFFFF; // for 64 bits
    constexpr std::uint32_t shift_bits = 32;         // for 64 bits

    hash_ = 0;
    hash_ |= first_read_ & element_mask;
    hash_ |= (static_cast<std::size_t>(number_of_reads_) & element_mask) << shift_bits;
}

bool operator==(const IndexDescriptor& lhs, const IndexDescriptor& rhs)
{
    return lhs.first_read() == rhs.first_read() && lhs.number_of_reads() == rhs.number_of_reads();
}

bool operator!=(const IndexDescriptor& lhs, const IndexDescriptor& rhs)
{
    return !(lhs == rhs);
}

std::size_t IndexDescriptorHash::operator()(const IndexDescriptor& index_descriptor) const
{
    return index_descriptor.get_hash();
}

std::vector<IndexDescriptor> group_reads_into_indices(const io::FastaParser& parser,
                                                      const number_of_basepairs_t max_basepairs_per_index)
{
    std::vector<IndexDescriptor> index_descriptors;

    const number_of_reads_t total_number_of_reads              = parser.get_num_seqences();
    read_id_t first_read_in_current_index                      = 0;
    number_of_reads_t number_of_reads_in_current_index         = 0;
    number_of_basepairs_t number_of_basepairs_in_current_index = 0;
    for (read_id_t read_id = 0; read_id < total_number_of_reads; read_id++)
    {
        number_of_basepairs_t basepairs_in_this_read = get_size<number_of_basepairs_t>(parser.get_sequence_by_id(read_id).seq);
        if (basepairs_in_this_read + number_of_basepairs_in_current_index > max_basepairs_per_index)
        {
            // adding this sequence would lead to index_descriptor being larger than max_basepairs_per_index
            // save current index_descriptor and start a new one
            index_descriptors.push_back({first_read_in_current_index, number_of_reads_in_current_index});
            first_read_in_current_index          = read_id;
            number_of_reads_in_current_index     = 1;
            number_of_basepairs_in_current_index = basepairs_in_this_read;
        }
        else
        {
            // add this sequence to the current index_descriptor
            number_of_basepairs_in_current_index += basepairs_in_this_read;
            ++number_of_reads_in_current_index;
        }
    }

    // save last index_descriptor
    index_descriptors.push_back({first_read_in_current_index, number_of_reads_in_current_index});

    return index_descriptors;
}

} // namespace cudamapper
} // namespace claragenomics
