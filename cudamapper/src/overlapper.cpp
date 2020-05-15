/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <algorithm>
#include <cstddef>
#include <future>
#include <mutex>
#include <vector>

#include <claragenomics/cudamapper/overlapper.hpp>
#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/types.hpp>

#include "cudamapper_utils.hpp"

namespace
{
bool overlaps_mergable(const claraparabricks::genomeworks::cudamapper::Overlap o1, const claraparabricks::genomeworks::cudamapper::Overlap o2)
{
    bool relative_strands_forward = (o2.relative_strand == claraparabricks::genomeworks::cudamapper::RelativeStrand::Forward) && (o1.relative_strand == claraparabricks::genomeworks::cudamapper::RelativeStrand::Forward);
    bool relative_strands_reverse = (o2.relative_strand == claraparabricks::genomeworks::cudamapper::RelativeStrand::Reverse) && (o1.relative_strand == claraparabricks::genomeworks::cudamapper::RelativeStrand::Reverse);

    if (!(relative_strands_forward || relative_strands_reverse))
    {
        return false;
    }

    bool ids_match = (o1.query_read_id_ == o2.query_read_id_) && (o1.target_read_id_ == o2.target_read_id_);

    if (!ids_match)
    {
        return false;
    }

    int query_gap = (o2.query_start_position_in_read_ - o1.query_end_position_in_read_);
    int target_gap;

    // If the strands are reverse strands, the coordinates of the target strand overlaps will be decreasing
    // as those of the query increase. We therefore need to know wether this is a forward or reverse match
    // before calculating the gap between overlaps.
    if (relative_strands_reverse)
    {
        target_gap = (o1.target_start_position_in_read_ - o2.target_end_position_in_read_);
    }
    else
    {
        target_gap = (o2.target_start_position_in_read_ - o1.target_end_position_in_read_);
    }

    auto gap_ratio    = static_cast<float>(std::min(query_gap, target_gap)) / static_cast<float>(std::max(query_gap, target_gap));
    bool gap_ratio_ok = (gap_ratio > 0.8) || ((query_gap < 500) && (target_gap < 500)); //TODO make these user-configurable?
    return gap_ratio_ok;
}

// Reverse complement lookup table
static char complement_array[26] = {
    84, 66, 71, 68, 69,
    70, 67, 72, 73, 74,
    75, 76, 77, 78, 79,
    80, 81, 82, 83, 65,
    85, 86, 87, 88, 89, 90};

void reverse_complement(std::string& s, const std::size_t len)
{
    for (std::size_t i = 0; i < len / 2; ++i)
    {
        char tmp       = s[i];
        s[i]           = static_cast<char>(complement_array[static_cast<int>(s[len - 1 - i] - 65)]);
        s[len - 1 - i] = static_cast<char>(complement_array[static_cast<int>(tmp) - 65]);
    }
}
std::string string_slice(const std::string& s, const std::size_t start, const std::size_t end)
{
    return s.substr(start, end - start);
}

claraparabricks::genomeworks::cga_string_view_t string_view_slice(const claraparabricks::genomeworks::cga_string_view_t& s, const std::size_t start, const std::size_t end)
{
    return s.substr(start, end - start);
}

} // namespace

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

void Overlapper::post_process_overlaps(std::vector<Overlap>& overlaps)
{
    const auto num_overlaps = get_size(overlaps);
    bool in_fuse            = false;
    int fused_target_start;
    int fused_query_start;
    int fused_target_end;
    int fused_query_end;
    int num_residues = 0;
    Overlap prev_overlap;

    for (int i = 1; i < num_overlaps; i++)
    {
        prev_overlap                  = overlaps[i - 1];
        const Overlap current_overlap = overlaps[i];
        //Check if previous overlap can be merged into the current one
        if (overlaps_mergable(prev_overlap, current_overlap))
        {
            if (!in_fuse)
            { // Entering a new fuse
                num_residues      = prev_overlap.num_residues_ + current_overlap.num_residues_;
                in_fuse           = true;
                fused_query_start = prev_overlap.query_start_position_in_read_;
                fused_query_end   = current_overlap.query_end_position_in_read_;

                // If the relative strands are forward, then the target positions are increasing.
                // However, if the relative strands are in the reverse direction, the target
                // positions along the read are decreasing. When fusing, this needs to be accounted for
                // by the following checks.
                if (current_overlap.relative_strand == RelativeStrand::Forward)
                {
                    fused_target_start = prev_overlap.target_start_position_in_read_;
                    fused_target_end   = current_overlap.target_end_position_in_read_;
                }
                else
                {
                    fused_target_start = current_overlap.target_start_position_in_read_;
                    fused_target_end   = prev_overlap.target_end_position_in_read_;
                }
            }
            else
            {
                // Continuing a fuse, query end is always incremented, however whether we increment the target start or
                // end depends on whether the overlap is a reverse or forward strand overlap.
                num_residues += current_overlap.num_residues_;
                fused_query_end = current_overlap.query_end_position_in_read_;
                // Query end has been incrememnted. Increment target end or start
                // depending on whether the overlaps are reverse or forward matching.
                if (current_overlap.relative_strand == RelativeStrand::Forward)
                {
                    fused_target_end = current_overlap.target_end_position_in_read_;
                }
                else
                {
                    fused_target_start = current_overlap.target_start_position_in_read_;
                }
            }
        }
        else
        {
            if (in_fuse)
            { //Terminate the previous overlap fusion
                in_fuse                                      = false;
                Overlap fused_overlap                        = prev_overlap;
                fused_overlap.query_start_position_in_read_  = fused_query_start;
                fused_overlap.target_start_position_in_read_ = fused_target_start;
                fused_overlap.query_end_position_in_read_    = fused_query_end;
                fused_overlap.target_end_position_in_read_   = fused_target_end;
                fused_overlap.num_residues_                  = num_residues;
                overlaps.push_back(fused_overlap);
                num_residues = 0;
            }
        }
    }
    //Loop terminates in the middle of an overlap fuse - fuse the overlaps.
    if (in_fuse)
    {
        Overlap fused_overlap                        = prev_overlap;
        fused_overlap.query_start_position_in_read_  = fused_query_start;
        fused_overlap.target_start_position_in_read_ = fused_target_start;
        fused_overlap.query_end_position_in_read_    = fused_query_end;
        fused_overlap.target_end_position_in_read_   = fused_target_end;
        fused_overlap.num_residues_                  = num_residues;
        overlaps.push_back(fused_overlap);
    }
}

void Overlapper::extend_overlap_by_sequence_similarity(Overlap& overlap,
                                                       cga_string_view_t& query_sequence,
                                                       cga_string_view_t& target_sequence,
                                                       const std::int32_t extension,
                                                       const float required_similarity)
{

    const position_in_read_t query_head_rescue_size  = std::min(overlap.query_start_position_in_read_, static_cast<position_in_read_t>(extension));
    const position_in_read_t target_head_rescue_size = std::min(overlap.target_start_position_in_read_, static_cast<position_in_read_t>(extension));
    // Calculate the shortest sequence length and use this as the window for comparison.
    const position_in_read_t head_rescue_size = std::min(query_head_rescue_size, target_head_rescue_size);

    const position_in_read_t query_head_start  = overlap.query_start_position_in_read_ - head_rescue_size;
    const position_in_read_t target_head_start = overlap.target_start_position_in_read_ - head_rescue_size;

    cga_string_view_t query_head_sequence  = string_view_slice(query_sequence, query_head_start, overlap.query_start_position_in_read_);
    cga_string_view_t target_head_sequence = string_view_slice(target_sequence, target_head_start, overlap.target_start_position_in_read_);

    float head_similarity = sequence_jaccard_similarity(query_head_sequence, target_head_sequence, 15, 1);
    if (head_similarity >= required_similarity)
    {
        overlap.query_start_position_in_read_  = overlap.query_start_position_in_read_ - head_rescue_size;
        overlap.target_start_position_in_read_ = overlap.target_start_position_in_read_ - head_rescue_size;
    }

    const position_in_read_t query_tail_rescue_size  = std::min(static_cast<position_in_read_t>(extension), static_cast<position_in_read_t>(query_sequence.length()) - overlap.query_end_position_in_read_);
    const position_in_read_t target_tail_rescue_size = std::min(static_cast<position_in_read_t>(extension), static_cast<position_in_read_t>(target_sequence.length()) - overlap.target_end_position_in_read_);
    // Calculate the shortest sequence length at the tail and use this as the window for comparison.
    const position_in_read_t tail_rescue_size = std::min(query_tail_rescue_size, target_tail_rescue_size);

    cga_string_view_t query_tail_sequence  = string_view_slice(query_sequence, overlap.query_end_position_in_read_, overlap.query_end_position_in_read_ + tail_rescue_size);
    cga_string_view_t target_tail_sequence = string_view_slice(target_sequence, overlap.target_end_position_in_read_, overlap.target_end_position_in_read_ + tail_rescue_size);

    const float tail_similarity = sequence_jaccard_similarity(query_tail_sequence, target_tail_sequence, 15, 1);
    if (tail_similarity >= required_similarity)
    {
        overlap.query_end_position_in_read_  = overlap.query_end_position_in_read_ + tail_rescue_size;
        overlap.target_end_position_in_read_ = overlap.target_end_position_in_read_ + tail_rescue_size;
    }

    //     std::cerr <<
    //     "head sz:" << head_rescue_size << " " <<
    //     query_head_sequence << " " <<
    //     target_head_sequence << " " <<
    //      "head sim: " << head_similarity << " " <<
    //       "tail sim: " << tail_similarity <<
    //        std::endl;
    // std::cerr <<
    //     "tail sz:" << tail_rescue_size << " " <<
    //     query_tail_sequence << " " <<
    //     target_tail_sequence << " " <<
    //      "tail sim: " << tail_similarity << " " <<
    //        std::endl;
}

void Overlapper::rescue_overlap_ends(std::vector<Overlap>& overlaps,
                                     const io::FastaParser& query_parser,
                                     const io::FastaParser& target_parser,
                                     const std::int32_t extension,
                                     const float required_similarity)
{

    auto reverse_overlap = [](cudamapper::Overlap& overlap, std::uint32_t target_sequence_length) {
        overlap.relative_strand      = overlap.relative_strand == RelativeStrand::Forward ? RelativeStrand::Reverse : RelativeStrand::Forward;
        position_in_read_t start_tmp = overlap.target_start_position_in_read_;
        // Oddly, the target_length_ field appears to be zero up till this point, so use the sequence's length instead.
        overlap.target_start_position_in_read_ = target_sequence_length - overlap.target_end_position_in_read_;
        overlap.target_end_position_in_read_   = target_sequence_length - start_tmp;
    };

    // Loop over all overlaps
    // For each overlap, retrieve the read sequence and
    // check the similarity of the overlapping head and tail sections (matched for length)
    // If they are more than or equal to <required_similarity> similar, extend the overlap start/end fields by <extension> basepairs.

    for (auto& overlap : overlaps)
    {
        // Track whether the overlap needs to be reversed from its original orientation on the '-' strand.
        bool reversed = false;

        // Overlap rescue at "head" (i.e., "left-side") of overlap
        // Get the sequences of the query and target
        const std::string query_sequence = query_parser.get_sequence_by_id(overlap.query_read_id_).seq;
        cga_string_view_t query_view(query_sequence);
        // target_sequence is non-const as it may be modified when reversing an overlap.
        std::string target_sequence = target_parser.get_sequence_by_id(overlap.target_read_id_).seq;

        if (overlap.relative_strand == RelativeStrand::Reverse)
        {

            reverse_overlap(overlap, static_cast<uint32_t>(target_sequence.length()));
            reverse_complement(target_sequence, target_sequence.length());
            reversed = true;
        }
        cga_string_view_t target_view(target_sequence);

        extend_overlap_by_sequence_similarity(overlap, query_view, target_view, 100, 0.9);

        if (reversed)
        {
            reverse_overlap(overlap, static_cast<uint32_t>(target_sequence.length()));
        }
    }
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
