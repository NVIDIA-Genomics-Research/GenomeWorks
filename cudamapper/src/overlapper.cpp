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
#include <cassert>
#include <claragenomics/io/fasta_parser.hpp>
#include "claragenomics/cudamapper/overlapper.hpp"
#include "claragenomics/cudaaligner/aligner.hpp"
#include "claragenomics/cudaaligner/alignment.hpp"
#include <iostream>

namespace claragenomics
{
namespace cudamapper
{

std::string get_query_from_overlap(Overlap const& overlap, claragenomics::io::FastaParser const& parser)
{
    claragenomics::io::FastaSequence s = parser.get_sequence_by_name(overlap.query_read_name_);
    if (overlap.query_start_position_in_read_ >= s.seq.size() || overlap.query_end_position_in_read_ >= s.seq.size())
        throw std::runtime_error("Overlap expected a longer FastaSequence.");
    std::string r;
    r.reserve(overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_);
    std::copy(begin(s.seq) + overlap.query_start_position_in_read_, begin(s.seq) + overlap.query_end_position_in_read_, std::back_inserter(r));
    return r;
}

std::string get_target_from_overlap(Overlap const& overlap, claragenomics::io::FastaParser const& parser)
{
    claragenomics::io::FastaSequence s = parser.get_sequence_by_name(overlap.target_read_name_);
    if (overlap.target_start_position_in_read_ >= s.seq.size() || overlap.target_end_position_in_read_ >= s.seq.size())
        throw std::runtime_error("Overlap expected a longer FastaSequence.");
    std::string r;
    r.reserve(overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_);
    std::copy(begin(s.seq) + overlap.target_start_position_in_read_, begin(s.seq) + overlap.target_end_position_in_read_, std::back_inserter(r));
    return r;
}

void Overlapper::filter_overlaps(std::vector<Overlap>& overlaps, size_t min_residues, size_t min_overlap_len)
{
    auto invalid_overlap = [&min_residues, &min_overlap_len](Overlap overlap) { return !((overlap.num_residues_ >= min_residues) &&
                                                                                         ((overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_) > min_overlap_len)); };

    auto newend = std::remove_if(overlaps.begin(), overlaps.end(), invalid_overlap);
    overlaps.erase(newend, overlaps.end());
}

void Overlapper::generate_alignments(std::vector<Overlap>& overlaps, claragenomics::io::FastaParser const& query_parser, claragenomics::io::FastaParser const& target_parser)
{
    using std::tie;
    // TODO remove these hardcoded default values
    int32_t device_id = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int32_t max_query_length  = 0;
    int32_t max_target_length = 0;
    for (auto const& o : overlaps)
    {
        max_query_length  = std::max<int32_t>(max_query_length, o.query_end_position_in_read_ - o.query_start_position_in_read_);
        max_target_length = std::max<int32_t>(max_target_length, o.target_end_position_in_read_ - o.target_start_position_in_read_);
    }

    std::cerr << "Total alignments " << overlaps.size() << std::endl;

    int32_t batch_size = 3072;

    std::unique_ptr<cudaaligner::Aligner> aligner = cudaaligner::create_aligner(max_query_length, max_target_length, batch_size, cudaaligner::global_alignment, stream, device_id);

    int32_t i         = 0;
    auto it           = begin(overlaps);
    int32_t processed = 0;
    while (it != end(overlaps))
    {
        std::string query  = get_query_from_overlap(*it, query_parser);
        std::string target = get_target_from_overlap(*it, target_parser);
        aligner->add_alignment(query.data(), query.size(), target.data(), target.size());

        if (i == batch_size)
        {
            aligner->align_all();
            aligner->sync_alignments();
            std::vector<std::shared_ptr<cudaaligner::Alignment>> alignments = aligner->get_alignments();
            if (alignments.size() != batch_size)
                std::cerr << "SOMETHING IS WRONGGG!" << std::endl;
            for (int32_t j = 0; j < batch_size; ++j)
            {
                (it - batch_size + j)->cigar_ = alignments[j]->convert_to_cigar();
            }
            processed += batch_size;
            std::cerr << "Processed " << processed << std::endl;
            i = 0;
            continue;
        }
        ++it;
        ++i;
    }
    aligner->align_all();
    aligner->sync_alignments();

    std::vector<std::shared_ptr<cudaaligner::Alignment>> alignments = aligner->get_alignments();
    for (int32_t j = 0; j < i; ++j)
    {
        (it - i + j)->cigar_ = alignments[j]->convert_to_cigar();
    }

    cudaStreamDestroy(stream);
}

void Overlapper::print_paf(const std::vector<Overlap>& overlaps)
{
    for (const auto& overlap : overlaps)
    {
        // Add basic overlap information.
        std::printf("%s\t%i\t%i\t%i\t%c\t%s\t%i\t%i\t%i\t%i\t%i\t%i",
                    overlap.query_read_name_.c_str(),
                    overlap.query_length_,
                    overlap.query_start_position_in_read_,
                    overlap.query_end_position_in_read_,
                    static_cast<unsigned char>(overlap.relative_strand),
                    overlap.target_read_name_.c_str(),
                    overlap.target_length_,
                    overlap.target_start_position_in_read_,
                    overlap.target_end_position_in_read_,
                    overlap.num_residues_,
                    0,
                    255);
        // If CIGAR string is generated, output in PAF.
        if (overlap.cigar_ != "")
        {
            std::printf("\tcg:Z:%s", overlap.cigar_.c_str());
        }
        // Add new line to demarcate new entry.
        std::printf("\n");
    }
}
} // namespace cudamapper
} // namespace claragenomics
