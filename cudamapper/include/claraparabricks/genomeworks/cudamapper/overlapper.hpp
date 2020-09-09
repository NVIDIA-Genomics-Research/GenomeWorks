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

#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

#include <thrust/execution_policy.h>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
/// \addtogroup cudamapper
/// \{

namespace details
{
namespace overlapper
{

// void filter_self_mappings(std::vector<Overlap>& overlaps,
//                           const io::FastaParser& query_parser,
//                           const io::FastaParser& target_parser,
//                           const double max_percent_similarity);

void filter_self_mappings(std::vector<Overlap>& overlaps,
                          const io::FastaParser& query_parser,
                          const io::FastaParser& target_parser,
                          const double max_percent_overlap);

/// \brief Extends a single overlap at its ends if the similarity of the query and target sequences is above a specified threshold.
/// \param overlap An Overlap which is modified in place. Any of the query_start_position_in_read, query_end_position_in_read,
/// target_start_position_in_read, and target_end_position_in_read fields may be modified.
/// \param query_sequence A std::string_view of the query read sequence.
/// \param target_sequence A std::string_view of the target read sequence.
/// \param extension The number of bases to extend at the head and tail of the overlap. If the head or tail is shorter than extension,
/// the function only tries to extend to the end of the read.
/// \param required_similarity The minimum similarity to require to extend an overlap.
void extend_overlap_by_sequence_similarity(Overlap& overlap,
                                           gw_string_view_t& query_sequence,
                                           gw_string_view_t& target_sequence,
                                           std::int32_t extension,
                                           float required_similarity);
///
/// \brief Removes overlaps from a vector (modifying in place) based on a boolean mask.
/// \param overlaps A vector (reference) of overlaps
/// \param mask A vector of bools the same length as overlaps. If an index is true, the overlap at the corresponding index in overlaps is removed.
///
void drop_overlaps_by_mask(std::vector<claraparabricks::genomeworks::cudamapper::Overlap>& overlaps, const std::vector<bool>& mask);

} // namespace overlapper
} // namespace details

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
    /// \param all_to_all True if the target and query indexes are of the same FASTx file. If true, ignore self-self mappings when retrieving overlaps.
    /// \param min_residues smallest number of residues (anchors) for an overlap to be accepted
    /// \param min_overlap_len the smallest overlap distance which is accepted
    /// \param min_bases_per_residue the minimum number of nucleotides per residue (e.g minimizer) in an overlap
    /// \param min_overlap_fraction the minimum ratio between the shortest and longest of the target and query components of an overlap. e.g if Query range is (150,1000) and target range is (1000,2000) then overlap fraction is 0.85
    virtual void get_overlaps(std::vector<Overlap>& fused_overlaps,
                              const device_buffer<Anchor>& d_anchors,
                              bool all_to_all,
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
    /// \param drop_fused_overlaps If true, remove overlaps that are fused into larger overlaps in output.
    static void post_process_overlaps(std::vector<Overlap>& overlaps, bool drop_fused_overlaps = false);

    /// \brief Given a vector of overlaps, extend the start/end of the overlaps based on the sequence similarity of the query and target.
    /// \param overlaps A vector of overlaps. This is modified in-place; query_start_position_in_read_, query_end_position_in_read_,
    /// target_start_position_in_read_ and target_end_position_in_read_ may be modified.
    /// \param query_parser A FastaParser for query sequences.
    /// \param target_parser A FastaParser for target sequences.
    /// \param extension The number of basepairs to extend and overlap.
    /// \param required_similarity The minimum similarity required to extend an overlap.
    static void rescue_overlap_ends(std::vector<Overlap>& overlaps,
                                    const io::FastaParser& query_parser,
                                    const io::FastaParser& target_parser,
                                    std::int32_t extension,
                                    float required_similarity);

    /// \brief Creates a Overlapper object
    /// \param allocator The device memory allocator to use for buffer allocations
    /// \param cuda_stream CUDA stream on which the work is to be done. Device arrays are also associated with this stream and will not be freed at least until all work issued on this stream before calling their destructor is done
    /// \return Instance of Overlapper
    static std::unique_ptr<Overlapper> create_overlapper(DefaultDeviceAllocator allocator,
                                                         const cudaStream_t cuda_stream = 0);
};
//}
} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
