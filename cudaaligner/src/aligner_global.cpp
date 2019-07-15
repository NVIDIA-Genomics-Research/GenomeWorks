/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cstring>
#include <algorithm>

#include <cuda_runtime_api.h>

#include <utils/signed_integer_utils.hpp>

#include "aligner_global.hpp"
#include "alignment_impl.hpp"
#include <cudautils/cudautils.hpp>
#include <logging/logging.hpp>

namespace claragenomics
{

namespace cudaaligner
{

constexpr int32_t calc_max_result_length(int32_t max_query_length, int32_t max_subject_length)
{
    constexpr int32_t alignment_bytes = 4;
    const int32_t max_length          = max_query_length + max_subject_length;
    return max_length + max_length % alignment_bytes;
}

AlignerGlobal::AlignerGlobal(int32_t max_query_length, int32_t max_subject_length, int32_t max_alignments, cudaStream_t stream, int32_t device_id)
    : max_query_length_(throw_on_negative(max_query_length, "max_query_length must be non-negative."))
    , max_subject_length_(throw_on_negative(max_subject_length, "max_subject_length must be non-negative."))
    , max_alignments_(throw_on_negative(max_alignments, "max_alignments must be non-negative."))
    , alignments_()
    , sequences_d_(2 * std::max(max_query_length, max_subject_length) * max_alignments, device_id)
    , sequences_h_(2 * std::max(max_query_length, max_subject_length) * max_alignments, device_id)
    , sequence_lengths_d_(2 * max_alignments, device_id)
    , sequence_lengths_h_(2 * max_alignments, device_id)
    , results_d_(calc_max_result_length(max_query_length, max_subject_length) * max_alignments, device_id)
    , results_h_(calc_max_result_length(max_query_length, max_subject_length) * max_alignments, device_id)
    , result_lengths_d_(max_alignments, device_id)
    , result_lengths_h_(max_alignments, device_id)
    , stream_(stream)
    , device_id_(device_id)
{
    if (max_alignments < 1)
    {
        throw std::runtime_error("Max alignments must be at least 1.");
    }
}

StatusType AlignerGlobal::add_alignment(const char* query, int32_t query_length, const char* subject, int32_t subject_length)
{
    if (query_length < 0 || subject_length < 0)
    {
        CGA_LOG_DEBUG("{} {}", "Negative subject or query length is not allowed.");
        return StatusType::generic_error;
    }

    int32_t const max_alignment_length = std::max(max_query_length_, max_subject_length_);
    int32_t const num_alignments       = get_size(alignments_);
    if (num_alignments >= max_alignments_)
    {
        CGA_LOG_DEBUG("{} {}", "Exceeded maximum number of alignments allowed : ", max_alignments_);
        return StatusType::exceeded_max_alignments;
    }

    if (query_length > max_query_length_)
    {
        CGA_LOG_DEBUG("{} {}", "Exceeded maximum length of query allowed : ", max_query_length_);
        return StatusType::exceeded_max_length;
    }

    if (subject_length > max_subject_length_)
    {
        CGA_LOG_DEBUG("{} {}", "Exceeded maximum length of subject allowed : ", max_subject_length_);
        return StatusType::exceeded_max_length;
    }

    memcpy(&sequences_h_[(2 * num_alignments) * max_alignment_length],
           query,
           sizeof(char) * query_length);
    memcpy(&sequences_h_[(2 * num_alignments + 1) * max_alignment_length],
           subject,
           sizeof(char) * subject_length);

    sequence_lengths_h_[2 * num_alignments]     = query_length;
    sequence_lengths_h_[2 * num_alignments + 1] = subject_length;

    std::shared_ptr<AlignmentImpl> alignment = std::make_shared<AlignmentImpl>(query,
                                                                               query_length,
                                                                               subject,
                                                                               subject_length);
    alignment->set_alignment_type(AlignmentType::global);
    alignments_.push_back(alignment);

    return StatusType::success;
}

StatusType AlignerGlobal::align_all()
{
    const int32_t max_alignment_length = std::max(max_query_length_, max_subject_length_);
    const int32_t num_alignments       = get_size(alignments_);
    const int32_t max_result_length    = calc_max_result_length(max_query_length_, max_subject_length_);
    CGA_CU_CHECK_ERR(cudaSetDevice(device_id_));
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(sequence_lengths_d_.data(),
                                     sequence_lengths_h_.data(),
                                     2 * sizeof(int32_t) * num_alignments,
                                     cudaMemcpyHostToDevice,
                                     stream_));
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(sequences_d_.data(),
                                     sequences_h_.data(),
                                     2 * sizeof(char) * max_alignment_length * num_alignments,
                                     cudaMemcpyHostToDevice,
                                     stream_));

    // Run kernel
    run_alignment(results_d_.data(), result_lengths_d_.data(),
                  max_result_length, sequences_d_.data(), sequence_lengths_d_.data(), sequence_lengths_h_.data(),
                  max_alignment_length,
                  num_alignments,
                  stream_);

    CGA_CU_CHECK_ERR(cudaMemcpyAsync(results_h_.data(),
                                     results_d_.data(),
                                     sizeof(int8_t) * max_result_length * num_alignments,
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(result_lengths_h_.data(),
                                     result_lengths_d_.data(),
                                     sizeof(int32_t) * num_alignments,
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    return StatusType::success;
}

StatusType AlignerGlobal::sync_alignments()
{
    CGA_CU_CHECK_ERR(cudaSetDevice(device_id_));
    CGA_CU_CHECK_ERR(cudaStreamSynchronize(stream_));

    const int32_t n_alignments      = get_size(alignments_);
    const int32_t max_result_length = calc_max_result_length(max_query_length_, max_subject_length_);
    std::vector<AlignmentState> al_state;
    for (int32_t i = 0; i < n_alignments; ++i)
    {
        al_state.clear();
        const int8_t* r_begin = results_h_.data() + i * max_result_length;
        const int8_t* r_end   = r_begin + result_lengths_h_[i];
        std::transform(r_begin, r_end, std::back_inserter(al_state), [](int8_t x) { return static_cast<AlignmentState>(x); });
        std::reverse(begin(al_state), end(al_state));
        AlignmentImpl* alignment = dynamic_cast<AlignmentImpl*>(alignments_[i].get());
        alignment->set_alignment(al_state);
        alignment->set_status(StatusType::success);
    }
    return StatusType::success;
}

void AlignerGlobal::reset()
{
    alignments_.clear();
}
} // namespace cudaaligner
} // namespace claragenomics
