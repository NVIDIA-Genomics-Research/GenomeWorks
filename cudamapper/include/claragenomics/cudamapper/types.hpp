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
#include <sys/resource.h>
#include <assert.h>

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
    std::vector<float> indexer_time;
    /// time (msecs) spent to complete matcher for each benchmark iteration
    std::vector<float> matcher_time;
    /// time (msecs) spent to complete overlapper for each benchmark iteration
    std::vector<float> overlapper_time;

    /// keep track of max device memory (GB) per benchmark iteration
    std::vector<double> device_mem;
    /// keep track of max host memory per (GB) benchmark iteration
    std::vector<double> host_mem;

private:
    /// time stamps to keep track of the elapsed time in msec
    rusage start_;
    rusage stop_;
    /// a debug flag to ensure start_timer is called before any stop_timer usage
    bool timer_initilized_ = false;
    /// accumulative compute time per benchmark iteration for indexer
    float index_time_ = 0.f;
    /// accumulative compute time per benchmark iteration for matcher
    float match_time_ = 0.f;
    /// accumulative compute time per benchmark iteration for overlapper
    float overlap_time_ = 0.f;
    /// max used RAM in (GB)
    float host_mem_ = 0.f;
    /// max used memory on GPU (GB)
    float device_mem_ = 0.f;

public:
    void start_timer()
    {
        getrusage(RUSAGE_SELF, &start_);
        timer_initilized_ = true;
    }

    // will update elapsed time between start_timing and stop_timing interval in msec
    void stop_timer_and_gather_data(char x)
    {
        // start_timer was not used before calling stop_timer
        assert(timer_initilized_ == true);
        getrusage(RUSAGE_SELF, &stop_);
        timer_initilized_ = false;
        float sec = stop_.ru_utime.tv_sec - start_.ru_utime.tv_sec;
        float usec = stop_.ru_utime.tv_usec - start_.ru_utime.tv_usec;
        float elapsed_time = (sec * 1000) + (usec / 1000);

        host_mem_ = std::max(host_mem_, (float)(stop_.ru_maxrss)/1000000.f);

        switch (x)
        {
            case 'i':
                index_time_ += elapsed_time;
                break;
            case 'm':
                match_time_ += elapsed_time;
                break;
            case 'o':
                overlap_time_ += elapsed_time;
                break;
            default:
                assert(false);
        }
    }
};

} // namespace cudamapper

} // namespace claragenomics
