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
#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/cudamapper/overlapper.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <mutex>
#include <future>
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

void Overlapper::update_read_names(std::vector<Overlap>& overlaps,
                                   const io::FastaParser& query_parser,
                                   const io::FastaParser& target_parser)
{
#pragma omp parallel for
    for (size_t i = 0; i < overlaps.size(); i++)
    {
        auto& o                             = overlaps[i];
        const std::string& query_read_name  = query_parser.get_sequence_by_id(o.query_read_id_).name;
        const std::string& target_read_name = target_parser.get_sequence_by_id(o.target_read_id_).name;

        o.query_read_name_ = new char[query_read_name.length() + 1];
        strcpy(o.query_read_name_, query_read_name.c_str());

        o.target_read_name_ = new char[target_read_name.length() + 1];
        strcpy(o.target_read_name_, target_read_name.c_str());

        o.query_length_  = query_parser.get_sequence_by_id(o.query_read_id_).seq.length();
        o.target_length_ = target_parser.get_sequence_by_id(o.target_read_id_).seq.length();
    }
}

namespace
{
} // namespace

void Overlapper::print_paf(const std::vector<Overlap>& overlaps, const std::vector<std::string>& cigar, const int k)
{
    int32_t idx = 0;
    for (const auto& overlap : overlaps)
    {
        // Add basic overlap information.
        std::printf("%s\t%i\t%i\t%i\t%c\t%s\t%i\t%i\t%i\t%i\t%ld\t%i",
                    overlap.query_read_name_,
                    overlap.query_length_,
                    overlap.query_start_position_in_read_,
                    overlap.query_end_position_in_read_,
                    static_cast<unsigned char>(overlap.relative_strand),
                    overlap.target_read_name_,
                    overlap.target_length_,
                    overlap.target_start_position_in_read_,
                    overlap.target_end_position_in_read_,
                    overlap.num_residues_ * k, // Print out the number of residue matches multiplied by kmer size to get approximate number of matching bases
                    std::max(std::abs(static_cast<std::int64_t>(overlap.target_start_position_in_read_) - static_cast<std::int64_t>(overlap.target_end_position_in_read_)),
                             std::abs(static_cast<std::int64_t>(overlap.query_start_position_in_read_) - static_cast<std::int64_t>(overlap.query_end_position_in_read_))), //Approximate alignment length
                    255);
        // If CIGAR string is generated, output in PAF.
        if (cigar.size() != 0)
        {
            std::printf("\tcg:Z:%s", cigar[idx].c_str());
        }
        // Add new line to demarcate new entry.
        std::printf("\n");
        idx++;
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
