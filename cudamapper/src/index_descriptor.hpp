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

#include <claragenomics/types.hpp>
#include <claragenomics/io/fasta_parser.hpp>

namespace claragenomics
{
namespace cudamapper
{

/// IndexDescriptor - Every Index is defined by its first read and the number of reads
class IndexDescriptor
{
public:
    /// \brief constructor
    IndexDescriptor(read_id_t first_read, read_id_t number_of_reads);

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
    read_id_t number_of_reads() const;

    /// \brief returns hash value
    std::size_t get_hash() const;

private:
    /// \brief generates hash
    void generate_hash();

    /// first read in index
    read_id_t first_read_;
    /// number of reads in index
    read_id_t number_of_reads_;
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

/// \brief returns a list of IndexDescriptors in which the sum of basepairs of all reads in one IndexDescriptor is at most max_index_size
/// If a single read exceeds max_chunk_size it will be placed in its own IndexDescriptor.
///
/// \param parser parser to get the reads from
/// \param max_index_size the maximum number of basepairs in an IndexDescriptor
/// \return list of IndexDescriptors
std::vector<IndexDescriptor> group_reads_in_indices(const io::FastaParser& parser,
                                                    const number_of_basepairs_t max_index_size = 1000000);

} // namespace cudamapper
} // namespace claragenomics
