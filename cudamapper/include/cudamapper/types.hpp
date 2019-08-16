/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <cstdint>

namespace claragenomics {

/// ArrayBlock - points to a part of an array
///
/// Contains the index of the first element in the block and the number of elements
struct ArrayBlock {
    /// index of the first element of the block
    size_t first_element_;
    /// number of elements of the block
    std::uint32_t block_size_;
};

/// position_in_read_t
using position_in_read_t = std::uint32_t;
/// representation_t
using representation_t = std::uint64_t; // this depends on kmer size, in some cases could also be 32-bit
/// read_id_t
using read_id_t = std::uint64_t; // can this be 32-bit?

/// Anchor - represents one anchor
///
/// Anchor is a pair of two sketch elements with the same sketch element representation from different reads
struct Anchor{
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
typedef struct Overlap {
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
    /// query read name (e.g from FASTA)
    std::string query_read_name_;
    /// target read name (e.g from FASTA)
    std::string target_read_name_;
    /// Number of residues (e.g anchors) between the two reads
    std::uint32_t num_residues_ = 0;

    std::uint32_t query_length_ = 0;
    std::uint32_t target_length_ = 0;
    /// Whether the overlap is considered valid by the generating overlapper
    bool overlap_complete = false;
} Overlap;
}
