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

#pragma once

#include <claraparabricks/genomeworks/types.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

/// IndexDescriptor - Every Index is defined by its first read and the number of reads
class IndexDescriptor
{
public:
    /// \brief constructor
    IndexDescriptor(read_id_t first_read,
                    number_of_reads_t number_of_reads);

    /// \brief copy constructor
    IndexDescriptor(const IndexDescriptor&) = default;
    /// \brief copy assignment operator
    IndexDescriptor& operator=(const IndexDescriptor&) = default;
    /// \brief move constructor
    IndexDescriptor(IndexDescriptor&&) = default;
    /// \brief move assignment operator
    IndexDescriptor& operator=(IndexDescriptor&&) = default;
    /// \brief destructor
    ~IndexDescriptor() = default;

    /// \brief getter
    read_id_t first_read() const;

    /// \brief getter
    number_of_reads_t number_of_reads() const;

    /// \brief returns hash value
    std::size_t get_hash() const;

private:
    /// \brief generates hash
    void generate_hash();

    /// first read in index
    read_id_t first_read_;
    /// number of reads in index
    number_of_reads_t number_of_reads_;
    /// hash of this object
    std::size_t hash_;
};

/// \brief equality operator
bool operator==(const IndexDescriptor& lhs,
                const IndexDescriptor& rhs);

/// \brief inequality operator
bool operator!=(const IndexDescriptor& lhs,
                const IndexDescriptor& rhs);

/// IndexDescriptorHash - operator() calculates hash of a given IndexDescriptor
struct IndexDescriptorHash
{
    /// \brief caclulates hash of given IndexDescriptor
    std::size_t operator()(const IndexDescriptor& index_descriptor) const;
};

/// \brief returns a list of IndexDescriptors in which the sum of basepairs of all reads in one IndexDescriptor is at most max_basepairs_per_index
/// If a single read exceeds max_chunk_size it will be placed in its own IndexDescriptor.
///
/// \param parser parser to get the reads from
/// \param max_basepairs_per_index the maximum number of basepairs in an IndexDescriptor
/// \return list of IndexDescriptors
std::vector<IndexDescriptor> group_reads_into_indices(const io::FastaParser& parser,
                                                      number_of_basepairs_t max_basepairs_per_index = 1000000);

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
