/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claragenomics/cudamapper/overlapper.hpp>

#include <algorithm>
#include <vector>

#include <omp.h>

#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

namespace
{
bool overlaps_mergable(const claragenomics::cudamapper::Overlap o1, const claragenomics::cudamapper::Overlap o2)
{
    bool relative_strands_forward = (o2.relative_strand == claragenomics::cudamapper::RelativeStrand::Forward) && (o1.relative_strand == claragenomics::cudamapper::RelativeStrand::Forward);
    bool relative_strands_reverse = (o2.relative_strand == claragenomics::cudamapper::RelativeStrand::Reverse) && (o1.relative_strand == claragenomics::cudamapper::RelativeStrand::Reverse);

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
} // namespace

namespace claragenomics
{
namespace cudamapper
{

void Overlapper::print_paf(const std::vector<Overlap>& overlaps,
                           const std::vector<std::string>& cigar,
                           const io::FastaParser& query_parser,
                           const io::FastaParser& target_parser,
                           const std::int32_t kmer_size,
                           std::mutex& write_output_mutex)
{
    assert(!cigar.empty() || (overlaps.size() && cigar.size()));

#pragma omp parallel
    {
        std::size_t number_of_threads = omp_get_num_threads();
        // All overlaps are saved to a single vector of chars and that vector is then printed to output.
        // Writing overlaps directly to output would be inefficinet as all writes to output have to protected by a mutex.
        //
        // Allocate approximately 150 characters for each overlap which will be processed by this thread,
        // if more characters are needed buffer will be reallocated.
        std::vector<char> buffer(150 * overlaps.size() / number_of_threads);
        // characters written buffer so far
        std::int64_t chars_in_buffer = 0;

#pragma omp for
        for (std::size_t i = 0; i < overlaps.size(); ++i)
        {
            const std::string& query_read_name  = query_parser.get_sequence_by_id(overlaps[i].query_read_id_).name;
            const std::string& target_read_name = target_parser.get_sequence_by_id(overlaps[i].target_read_id_).name;

            // (over)estimate the number of character that are going to be needed
            // 150 is an overestimate of number of characters that are going to be needed for non-string values
            std::int32_t expected_chars = 150 + query_read_name.length() + target_read_name.length();
            if (!cigar.empty())
            {
                expected_chars += cigar[i].length();
            }

            // if there is not enough space in buffer reallocate
            if (buffer.size() - chars_in_buffer < expected_chars)
            {
                buffer.resize(buffer.size() * 2 + expected_chars);
            }

            // Add basic overlap information.
            std::int32_t added_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                                    "%s\t%lu\t%i\t%i\t%c\t%s\t%lu\t%i\t%i\t%i\t%ldem\t%i",
                                                    query_read_name.c_str(),
                                                    query_parser.get_sequence_by_id(overlaps[i].query_read_id_).seq.length(),
                                                    overlaps[i].query_start_position_in_read_,
                                                    overlaps[i].query_end_position_in_read_,
                                                    static_cast<unsigned char>(overlaps[i].relative_strand),
                                                    target_read_name.c_str(),
                                                    target_parser.get_sequence_by_id(overlaps[i].target_read_id_).seq.length(),
                                                    overlaps[i].target_start_position_in_read_,
                                                    overlaps[i].target_end_position_in_read_,
                                                    overlaps[i].num_residues_ * kmer_size, // Print out the number of residue matches multiplied by kmer size to get approximate number of matching bases
                                                    std::max(std::abs(static_cast<std::int64_t>(overlaps[i].target_start_position_in_read_) - static_cast<std::int64_t>(overlaps[i].target_end_position_in_read_)),
                                                             std::abs(static_cast<std::int64_t>(overlaps[i].query_start_position_in_read_) - static_cast<std::int64_t>(overlaps[i].query_end_position_in_read_))), //Approximate alignment length
                                                    255);
            chars_in_buffer += added_chars;

            // If CIGAR string is generated, output in PAF.
            if (!cigar.empty())
            {
                added_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                           "\tcg:Z:%s",
                                           cigar[i].c_str());
                chars_in_buffer += added_chars;
            }

            // Add new line to demarcate new entry.
            buffer.data()[chars_in_buffer] = '\n';
            ++chars_in_buffer;
        }
        buffer[chars_in_buffer] = '\0';

        std::lock_guard<std::mutex> lg(write_output_mutex);
        printf("%s", buffer.data());
    }
}

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
} // namespace cudamapper
} // namespace claragenomics
