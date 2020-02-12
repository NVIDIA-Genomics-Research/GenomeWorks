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
#include <claragenomics/cudaaligner/aligner.hpp>
#include <claragenomics/cudaaligner/alignment.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <mutex>
#include <future>

namespace claragenomics
{
namespace cudamapper
{

void Overlapper::filter_overlaps(std::vector<Overlap>& filtered_overlaps, const std::vector<Overlap>& overlaps, size_t min_residues, size_t min_overlap_len)
{
    auto valid_overlap = [&min_residues, &min_overlap_len](Overlap overlap) { return (
                                                                                  (overlap.num_residues_ >= min_residues) &&
                                                                                  ((overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_) > min_overlap_len) &&
                                                                                  !( // Reject overlaps where the query and target sections are exactly the same, otherwise miniasm has trouble.
                                                                                      (std::string(overlap.query_read_name_) == std::string(overlap.target_read_name_)) &&
                                                                                      overlap.query_start_position_in_read_ == overlap.target_start_position_in_read_ &&
                                                                                      overlap.query_end_position_in_read_ == overlap.target_end_position_in_read_)); };

    std::copy_if(overlaps.begin(), overlaps.end(),
                 std::back_inserter(filtered_overlaps),
                 valid_overlap);
}

void Overlapper::align_overlaps(std::vector<Overlap>& overlaps, const claragenomics::io::FastaParser& query_parser, const claragenomics::io::FastaParser& target_parser, std::vector<std::string>& cigar)
{
    int32_t max_query_size  = 0;
    int32_t max_target_size = 0;
    for (overlap : overlaps)
    {
        int32_t query_overlap_size  = overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_;
        int32_t target_overlap_size = overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_;
        if (query_overlap_size > max_query_size)
            max_query_size = query_overlap_size;
        if (target_overlap_size > max_target_size)
            max_target_size = target_overlap_size;
    }

    int32_t overlap_idx = 0;
    std::cerr << "Overlaps to align - " << overlaps.size() << std::endl;

    const float space_per_base = 0.000000027f; // Heuristic estimation of space per base in MB during CUDA alignment
    float space_per_alignment     = space_per_base * max_query_size * max_target_size;
    size_t free, total;
    CGA_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
    free                   = free / (1024 * 1024);
    int32_t max_alignments = (free * 90 / 100) / space_per_alignment;
    int32_t streams               = 4;
    int32_t batch_size = max_alignments / streams;

    std::mutex overlap_idx_mtx;
    std::vector<std::future<void>> align_futures;
    for (int32_t t = 0; t < streams; t++)
    {
        align_futures.push_back(std::async(std::launch::async, [&overlap_idx_mtx, &overlaps, &query_parser, &target_parser, &overlap_idx, max_query_size, max_target_size, &cigar, batch_size]() {
            int32_t device_id;
            CGA_CU_CHECK_ERR(cudaGetDevice(&device_id));
            cudaStream_t stream;
            CGA_CU_CHECK_ERR(cudaStreamCreate(&stream));
            std::unique_ptr<claragenomics::cudaaligner::Aligner> batch =
                claragenomics::cudaaligner::create_aligner(
                    max_query_size,
                    max_target_size,
                    batch_size,
                    claragenomics::cudaaligner::AlignmentType::global_alignment,
                    stream,
                    device_id);
            while (true)
            {
                int32_t idx_start, idx_end = 0;
                // Get the range of overlaps for this batch
                {
                    std::lock_guard<std::mutex> lck(overlap_idx_mtx);
                    if (overlap_idx == overlaps.size())
                    {
                        break;
                    }
                    else
                    {
                        idx_start   = overlap_idx;
                        idx_end     = std::min(idx_start + batch_size, static_cast<int32_t>(overlaps.size()));
                        overlap_idx = idx_end;
                    }
                }
                for (int32_t idx = idx_start; idx < idx_end; idx++)
                {
                    Overlap& overlap                              = overlaps[idx];
                    const claragenomics::io::FastaSequence query  = query_parser.get_sequence_by_id(overlap.query_read_id_);
                    const claragenomics::io::FastaSequence target = target_parser.get_sequence_by_id(overlap.target_read_id_);
                    const char* query_start                       = &query.seq[overlap.query_start_position_in_read_];
                    int32_t query_length                          = overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_;
                    const char* target_start                      = &target.seq[overlap.target_start_position_in_read_];
                    int32_t target_length                         = overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_;
                    claragenomics::cudaaligner::StatusType status = batch->add_alignment(query_start, query_length, target_start, target_length);
                    if (status != claragenomics::cudaaligner::success)
                    {
                        throw std::runtime_error("Experienced error type " + std::to_string(status));
                    }
                }
                // Launch alignment on the GPU. align_all is an async call.
                batch->align_all();
                // Synchronize all alignments.
                batch->sync_alignments();
                const std::vector<std::shared_ptr<claragenomics::cudaaligner::Alignment>>& alignments = batch->get_alignments();
                {
                    CGA_NVTX_RANGE(profiler, "copy_alignments");
                    int32_t counter = 0;
                    for (const auto& alignment : alignments)
                    {
                        cigar[idx_start + counter] = alignment->convert_to_cigar();
                        counter++;
                    }
                }
                // Reset batch to reuse memory for new alignments.
                batch->reset();
            }
            CGA_CU_CHECK_ERR(cudaStreamDestroy(stream));
        }));
    }

    for (auto& f : align_futures)
    {
        f.get();
    }
}

void Overlapper::print_paf(const std::vector<Overlap>& overlaps, const std::vector<std::string>& cigar)
{
    int32_t idx = 0;
    for (const auto& overlap : overlaps)
    {
        // Add basic overlap information.
        std::printf("%s\t%i\t%i\t%i\t%c\t%s\t%i\t%i\t%i\t%i\t%i\t%i",
                    overlap.query_read_name_,
                    overlap.query_length_,
                    overlap.query_start_position_in_read_,
                    overlap.query_end_position_in_read_,
                    static_cast<unsigned char>(overlap.relative_strand),
                    overlap.target_read_name_,
                    overlap.target_length_,
                    overlap.target_start_position_in_read_,
                    overlap.target_end_position_in_read_,
                    overlap.num_residues_,
                    0,
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
} // namespace cudamapper
} // namespace claragenomics
