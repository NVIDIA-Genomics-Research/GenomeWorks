/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

namespace claraparabricks
{

namespace genomeworks
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

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
