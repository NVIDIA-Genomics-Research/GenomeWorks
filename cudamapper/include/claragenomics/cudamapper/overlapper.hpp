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

#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/cudamapper/types.hpp>

#include <thrust/execution_policy.h>

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
    virtual void get_overlaps(std::vector<Overlap>& fused_overlaps,
                              device_buffer<Anchor>& d_anchors,
                              size_t min_residues,
                              size_t min_overlap_len) = 0;

    /// \brief prints overlaps to stdout in <a href="https://github.com/lh3/miniasm/blob/master/PAF.md">PAF format</a>
    static void print_paf(const std::vector<Overlap>& overlaps);

    /// \brief removes overlaps which are unlikely to be true overlaps
    /// \param filtered_overlaps Output vector in which to place filtered overlaps
    /// \param overlaps vector of Overlap objects to be filtered
    /// \param min_residues smallest number of residues (anchors) for an overlap to be accepted
    /// \param min_overlap_len the smallest overlap distance which is accepted
    static void filter_overlaps(std::vector<Overlap>& filtered_overlaps,
                                const std::vector<Overlap>& overlaps,
                                size_t min_residues    = 20,
                                size_t min_overlap_len = 50);

    /// \brief updates read names for vector of overlaps output from get_overlaps
    /// \param overlaps input vector of overlaps generated in get_overlaps
    /// \param index_query
    /// \param index_target
    static void update_read_names(std::vector<Overlap>& overlaps,
                                  const Index& index_query,
                                  const Index& index_target);
};
//}
} // namespace cudamapper

} // namespace claragenomics
