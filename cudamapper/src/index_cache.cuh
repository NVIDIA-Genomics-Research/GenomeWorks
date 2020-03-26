/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claragenomics/cudamapper/types.hpp>

namespace claragenomics
{
namespace cudamapper
{

/// IndexDescriptor - Every Index is defined by its first read and the number of reads
struct IndexDescriptor
{
    /// first read in index
    read_id_t first_read;
    /// number of reads in index
    read_id_t number_of_reads;
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

} // namespace cudamapper
} // namespace claragenomics