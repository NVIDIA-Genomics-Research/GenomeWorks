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

#include <vector>

#include <claragenomics/cudamapper/types.hpp>
#include <claragenomics/cudamapper/overlapper.hpp>
#include <claragenomics/utils/device_buffer.hpp>

namespace claragenomics
{
namespace cudamapper
{

/// OverlapperTriggered - generates overlaps and displays them on screen.
/// Uses a dynamic programming approach where an overlap is "triggered" when a run of
/// Anchors (e.g 3) with a score above a threshold is encountered and untriggerred
/// when a single anchor with a threshold below the value is encountered.
/// query_read_name_ and target_read_name_ in output overlaps are not initialized and
/// will be updated after Overlapper::update_read_names() call
class OverlapperTriggered : public Overlapper
{

public:
    /// \brief finds all overlaps
    /// Uses a dynamic programming approach where an overlap is "triggered" when a run of
    /// Anchors (e.g 3) with a score above a threshold is encountered and untriggerred
    /// when a single anchor with a threshold below the value is encountered.
    /// \param fused_overlaps Output vector into which generated overlaps will be placed, query_read_name_ and target_read_name_ for each vector entry remains null. They will be updated after Overlapper::update_read_names() call
    /// \param d_anchors vector of anchors sorted by query_read_id -> target_read_id -> query_position_in_read -> target_position_in_read (meaning sorted by query_read_id, then within a group of anchors with the same value of query_read_id sorted by target_read_id and so on)
    /// \param min_residues smallest number of residues (anchors) for an overlap to be accepted
    /// \param min_overlap_len the smallest overlap distance which is accepted
    /// \param min_bases_per_residue the minimum number of nucleotides per residue (e.g minimizer) in an overlap
    /// \param min_overlap_fraction the minimum ratio between the shortest and longest of the target and query components of an overlap. e.g if Query range is (150,1000) and target range is (1000,2000) then overlap fraction is 0.85
    /// \return vector of Overlap objects
    void get_overlaps(std::vector<Overlap>& fused_overlaps,
                      const device_buffer<Anchor>& d_anchors,
                      int64_t min_residues          = 20,
                      int64_t min_overlap_len       = 50,
                      int64_t min_bases_per_residue = 50,
                      float min_overlap_fraction    = 0.9) override;

    explicit OverlapperTriggered(DefaultDeviceAllocator,
                                 const cudaStream_t cuda_stream = 0);

private:
    DefaultDeviceAllocator _allocator;
    cudaStream_t _cuda_stream;
};
} // namespace cudamapper
} // namespace claragenomics
