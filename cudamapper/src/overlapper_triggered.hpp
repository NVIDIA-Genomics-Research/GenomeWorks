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
class OverlapperTriggered : public Overlapper
{

public:
    /// \brief finds all overlaps
    /// Uses a dynamic programming approach where an overlap is "triggered" when a run of
    /// Anchors (e.g 3) with a score above a threshold is encountered and untriggerred
    /// when a single anchor with a threshold below the value is encountered.
    /// \param overlaps Output vector into which generated overlaps will be placed
    /// \param anchors vector of anchors sorted by query_read_id -> target_read_id -> query_position_in_read -> target_position_in_read (meaning sorted by query_read_id, then within a group of anchors with the same value of query_read_id sorted by target_read_id and so on)
    /// \param index_query
    /// \param index_target
    /// \param min_residues smallest number of residues (anchors) for an overlap to be accepted
    /// \param min_overlap_len the smallest overlap distance which is accepted
    /// \return vector of Overlap objects
    void get_overlaps(std::vector<Overlap>& overlaps, device_buffer<Anchor>& anchors,
                      const Index& index_query, const Index& index_target,
                      size_t min_residues = 20, size_t min_overlap_len = 50) override;

    explicit OverlapperTriggered(std::shared_ptr<DeviceAllocator>);
    ~OverlapperTriggered();

private:
    std::shared_ptr<DeviceAllocator> _allocator;
    cudaStream_t stream;
};
} // namespace cudamapper
} // namespace claragenomics
