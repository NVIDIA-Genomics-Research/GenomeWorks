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

#include <cstdint>
#include <string>

#include <claraparabricks/genomeworks/types.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

/// ArrayBlock - points to a part of an array
///
/// Contains the index of the first element in the block and the number of elements
struct ArrayBlock
{
    /// index of the first element of the block
    size_t first_element_;
    /// number of elements of the block
    std::uint32_t block_size_;
};

/// representation_t
using representation_t = std::uint64_t; // this depends on kmer size, in some cases could also be 32-bit

/// Relative strand - represents whether query and target
/// are on the same DNA strand (i.e Forward) or not (i.e Reverse).
enum class RelativeStrand : unsigned char
{
    Forward = '+',
    Reverse = '-',
};

/// Anchor - represents one anchor
///
/// Anchor is a pair of two sketch elements with the same sketch element representation from different reads
struct Anchor
{
    /// read ID of query
    read_id_t query_read_id_;
    /// read ID of target
    read_id_t target_read_id_;
    /// position of first sketch element in query_read_id_
    position_in_read_t query_position_in_read_;
    /// position of second sketch element in target_read_id_
    position_in_read_t target_position_in_read_;
};

/// Overlap - represents one overlap between two substrings
///
/// Overlap is a region of two strings which is considered to be the same underlying biological sequence.
/// The overlapping region need not be identical across both substrings.
typedef struct Overlap
{
    /// internal read ID for query
    read_id_t query_read_id_;
    /// internal read ID for target
    read_id_t target_read_id_;
    /// start position in the query
    position_in_read_t query_start_position_in_read_;
    /// start position in the target
    position_in_read_t target_start_position_in_read_;
    /// end position in the query
    position_in_read_t query_end_position_in_read_;
    /// end position in the target
    position_in_read_t target_end_position_in_read_;
    /// Relative strand: Forward ("+") or Reverse("-")
    RelativeStrand relative_strand;
    /// Number of residues (e.g anchors) between the two reads
    std::uint32_t num_residues_ = 0;
    /// Whether the overlap is considered valid by the generating overlapper
    bool overlap_complete = false;

} Overlap;

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
