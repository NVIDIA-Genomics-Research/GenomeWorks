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

#include <claragenomics/cudamapper/types.hpp>
#include <claragenomics/utils/genomeutils.hpp>
#include <claragenomics/io/fasta_parser.hpp>

#include <thrust/execution_policy.h>
#include <claragenomics/utils/device_buffer.hpp>

namespace claragenomics
{

namespace cudamapper
{
/// \addtogroup cudamapper
/// \{

/// class Overlapper
/// Given anchors and a read index, calculates overlaps between reads
class Overlapper
{
public:
    /// \brief Virtual destructor for Overlapper
    virtual ~Overlapper() = default;

    /// \brief returns overlaps for a set of reads
    /// \param fused_overlaps Output vector into which generated overlaps will be placed
    /// \param d_anchors vector of anchors sorted by query_read_id -> target_read_id -> query_position_in_read -> target_position_in_read (meaning sorted by query_read_id, then within a group of anchors with the same value of query_read_id sorted by target_read_id and so on)
    /// \param min_residues smallest number of residues (anchors) for an overlap to be accepted
    /// \param min_overlap_len the smallest overlap distance which is accepted
    /// \param min_bases_per_residue the minimum number of nucleotides per residue (e.g minimizer) in an overlap
    /// \param min_overlap_fraction the minimum ratio between the shortest and longest of the target and query components of an overlap. e.g if Query range is (150,1000) and target range is (1000,2000) then overlap fraction is 0.85
    virtual void get_overlaps(std::vector<Overlap>& fused_overlaps,
                              const device_buffer<Anchor>& d_anchors,
                              int64_t min_residues,
                              int64_t min_overlap_len,
                              int64_t min_bases_per_residue,
                              float min_overlap_fraction) = 0;

    /// \brief removes overlaps which are unlikely to be true overlaps
    /// \param filtered_overlaps Output vector in which to place filtered overlaps
    /// \param overlaps vector of Overlap objects to be filtered
    /// \param min_residues smallest number of residues (anchors) for an overlap to be accepted
    /// \param min_overlap_len the smallest overlap distance which is accepted
    static void filter_overlaps(std::vector<Overlap>& filtered_overlaps,
                                const std::vector<Overlap>& overlaps,
                                int64_t min_residues    = 20,
                                int64_t min_overlap_len = 50);

    /// \brief Identified overlaps which can be combined into a larger overlap and add them to the input vector
    /// \param overlaps reference to vector of Overlaps. New overlaps (result of fusing) are added to this vector
    static void post_process_overlaps(std::vector<Overlap>& overlaps);

    ///
    ///@brief Given a vector of overlaps, extend the start/end of the overlaps based on the sequence similarity of the query and target.
    ///
    ///@param overlaps A vector of overlaps. This is modified in-place; query_start_position_in_read_, query_end_position_in_read_,
    /// target_start_position_in_read_ and target_end_position_in_read_ may be modified.
    ///@param query_parser A FastaParser for query sequences.
    ///@param target_parser A FastaParser for target sequences.
    ///@param max_extension The number of basepairs to extend and overlap.
    ///@param required_similarity The minimum similarity required to extend an overlap.
    static void rescue_overlap_ends(std::vector<Overlap>& overlaps,
                                    const io::FastaParser& query_parser,
                                    const io::FastaParser& target_parser,
                                    std::int32_t extension,
                                    float required_similarity);
};
//}
} // namespace cudamapper

} // namespace claragenomics
