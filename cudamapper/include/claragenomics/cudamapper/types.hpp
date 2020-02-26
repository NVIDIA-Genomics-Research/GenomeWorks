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
#include <string>
namespace claragenomics
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

/// position_in_read_t
using position_in_read_t = std::uint32_t;
/// representation_t
using representation_t = std::uint64_t; // this depends on kmer size, in some cases could also be 32-bit
/// read_id_t
using read_id_t = std::uint32_t;

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
    /// query read name (e.g from FASTA)
    char* query_read_name_ = nullptr;
    /// target read name (e.g from FASTA)
    char* target_read_name_ = nullptr;
    /// Relative strand: Forward ("+") or Reverse("-")
    RelativeStrand relative_strand;
    /// Number of residues (e.g anchors) between the two reads
    std::uint32_t num_residues_ = 0;
    /// Length of query sequence
    std::uint32_t query_length_ = 0;
    /// Length of target sequence
    std::uint32_t target_length_ = 0;
    /// Whether the overlap is considered valid by the generating overlapper
    bool overlap_complete = false;
    /// CIGAR string for alignment of mapped section.
    char* cigar_ = nullptr;

    //TODO add a destructor and copy constructor to remove need for this function
    /// \brief Free memory associated with Overlap.
    /// Since query_read_name_, target_read_name_ and cigar_ are char * types,
    /// they are not freed when Overlap is deleted.
    void clear()
    {
        delete[] target_read_name_;
        target_read_name_ = nullptr;

        delete[] query_read_name_;
        query_read_name_ = nullptr;

        delete[] cigar_;
        cigar_ = nullptr;
    }

} Overlap;

/// Data structure for storing benchmark data
///
/// contains vectors keep record of time and memory per benchmark iterations
/// a benchmark iteration refers to processing one batch of query indices
struct BenchMarkData
{
    /// time (msecs) spent to complete indexer for each benchmark iteration
    std::vector<double> indexer_time;
    /// time (msecs) spent to complete matcher for each benchmark iteration
    std::vector<double> matcher_time;
    /// time (msecs) spent to complete overlapper for each benchmark iteration
    std::vector<double> overlapper_time;
    /// total time (msecs) spent to complete each benchmark iteration
    std::vector<double> total_time;
    /// keep track of max device memory (GB) per benchmark iteration
    std::vector<double> device_mem;
    /// keep track of max host memory per (GB) benchmark iteration
    std::vector<double> host_mem;
};

} // namespace cudamapper

} // namespace claragenomics
