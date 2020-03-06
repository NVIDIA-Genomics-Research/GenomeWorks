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
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <mutex>
#include <future>

namespace claragenomics
{
namespace cudamapper
{

void Overlapper::update_read_names(std::vector<Overlap>& overlaps,
                                   const Index& index_query,
                                   const Index& index_target)
{
#pragma omp parallel for
    for (size_t i = 0; i < overlaps.size(); i++)
    {
        auto& o                             = overlaps[i];
        const std::string& query_read_name  = index_query.read_id_to_read_name(o.query_read_id_);
        const std::string& target_read_name = index_target.read_id_to_read_name(o.target_read_id_);

        o.query_read_name_ = new char[query_read_name.length() + 1];
        strcpy(o.query_read_name_, query_read_name.c_str());

        o.target_read_name_ = new char[target_read_name.length() + 1];
        strcpy(o.target_read_name_, target_read_name.c_str());

        o.query_length_  = index_query.read_id_to_read_length(o.query_read_id_);
        o.target_length_ = index_target.read_id_to_read_length(o.target_read_id_);
    }
}

void Overlapper::align_overlaps(std::vector<Overlap>& overlaps,
                                const claragenomics::io::FastaParser& query_parser,
                                const claragenomics::io::FastaParser& target_parser,
                                int32_t num_batches,
                                std::vector<std::string>& cigar)
{
    // Calculate max target/query size in overlaps
    int32_t max_query_size  = 0;
    int32_t max_target_size = 0;
    for (const auto& overlap : overlaps)
    {
        int32_t query_overlap_size  = overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_;
        int32_t target_overlap_size = overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_;
        if (query_overlap_size > max_query_size)
            max_query_size = query_overlap_size;
        if (target_overlap_size > max_target_size)
            max_target_size = target_overlap_size;
    }

    // Heuristically calculate max alignments possible with available memory based on
    // empirical measurements of memory needed for alignment per base.
    const float memory_per_base = 0.03f; // Estimation of space per base in bytes for alignment
    float memory_per_alignment  = memory_per_base * max_query_size * max_target_size;
    size_t free, total;
    CGA_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
    const size_t max_alignments = (static_cast<float>(free) * 85 / 100) / memory_per_alignment; // Using 85% of available memory
    int32_t batch_size          = std::min(get_size<int32_t>(overlaps), static_cast<int32_t>(max_alignments)) / num_batches;
    std::cerr << "Aligning " << overlaps.size() << " overlaps (" << max_query_size << "x" << max_target_size << ") with batch size " << batch_size << std::endl;

    int32_t overlap_idx = 0;
    std::mutex overlap_idx_mtx;

    // Compute alignments for overlaps
    auto compute_alignment_fn = [&overlap_idx_mtx, &overlaps, &query_parser, &target_parser, &overlap_idx, max_query_size, max_target_size, &cigar, batch_size]() {
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
            int32_t idx_start = 0, idx_end = 0;
            // Get the range of overlaps for this batch
            {
                std::lock_guard<std::mutex> lck(overlap_idx_mtx);
                if (overlap_idx == get_size<int32_t>(overlaps))
                {
                    break;
                }
                else
                {
                    idx_start   = overlap_idx;
                    idx_end     = std::min(idx_start + batch_size, get_size<int32_t>(overlaps));
                    overlap_idx = idx_end;
                }
            }
            for (int32_t idx = idx_start; idx < idx_end; idx++)
            {
                const Overlap& overlap                        = overlaps[idx];
                const claragenomics::io::FastaSequence query  = query_parser.get_sequence_by_id(overlap.query_read_id_);
                const claragenomics::io::FastaSequence target = target_parser.get_sequence_by_id(overlap.target_read_id_);
                const char* query_start                       = &query.seq[overlap.query_start_position_in_read_];
                const int32_t query_length                    = overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_;
                const char* target_start                      = &target.seq[overlap.target_start_position_in_read_];
                const int32_t target_length                   = overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_;
                claragenomics::cudaaligner::StatusType status = batch->add_alignment(query_start, query_length, target_start, target_length,
                                                                                     false, overlap.relative_strand == RelativeStrand::Reverse);
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
                for (int32_t i = 0; i < get_size<int32_t>(alignments); i++)
                {
                    cigar[idx_start + i] = alignments[i]->convert_to_cigar();
                }
            }
            // Reset batch to reuse memory for new alignments.
            batch->reset();
        }
        CGA_CU_CHECK_ERR(cudaStreamDestroy(stream));
    };

    // Launch alignment function in separate threads
    std::vector<std::future<void>> align_futures;
    for (int32_t t = 0; t < num_batches; t++)
    {
        align_futures.push_back(std::async(std::launch::async, compute_alignment_fn));
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
