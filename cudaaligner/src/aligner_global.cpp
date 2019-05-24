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

#define GW_LOG_LEVEL gw_log_level_warn

#include "aligner_global.hpp"
#include "alignment_impl.hpp"
#include "ukkonen_gpu.cuh"
#include <cudautils/cudautils.hpp>
#include <logging/logging.hpp>
#include "batched_device_matrices.cuh"

namespace genomeworks
{

namespace cudaaligner
{

static constexpr float max_target_query_length_difference = 0.1; // query has to be >=90% of target length

AlignerGlobal::AlignerGlobal(uint32_t max_query_length, uint32_t max_subject_length, uint32_t max_alignments, uint32_t device_id)
    : max_query_length_(max_query_length)
    , max_subject_length_(max_subject_length)
    , max_alignments_(max_alignments)
    , alignments_()
    , stream_(nullptr)
    , device_id_(device_id)
{
    // Must have at least one alignment.
    if (max_alignments < 1)
    {
        throw std::runtime_error("Max alignments must be at least 1.");
    }

    // Allocate buffers
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    GW_CU_CHECK_ERR(cudaMalloc((void**)&sequences_d_,
                               2 * sizeof(char) * std::max(max_query_length_, max_subject_length_) * max_alignments_));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**)&sequences_h_,
                                  2 * sizeof(char) * std::max(max_query_length_, max_subject_length_) * max_alignments_,
                                  cudaHostAllocDefault));

    GW_CU_CHECK_ERR(cudaMalloc((void**)&sequence_lengths_d_,
                               2 * sizeof(uint32_t) * max_alignments_));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**)&sequence_lengths_h_,
                                  2 * sizeof(uint32_t) * max_alignments_,
                                  cudaHostAllocDefault));

    GW_CU_CHECK_ERR(cudaMalloc((void**)&results_d_,
                               sizeof(uint8_t) * (max_query_length_ + max_subject_length_) * max_alignments_));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**)&results_h_,
                                  sizeof(uint8_t) * (max_query_length_ + max_subject_length_) * max_alignments_,
                                  cudaHostAllocDefault));

    GW_CU_CHECK_ERR(cudaMalloc((void**)&result_lengths_d_,
                               sizeof(uint32_t) * max_alignments_));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**)&result_lengths_h_,
                                  sizeof(uint32_t) * max_alignments_,
                                  cudaHostAllocDefault));
}

AlignerGlobal::~AlignerGlobal()
{
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    // Free up buffers
    GW_CU_CHECK_ERR(cudaFree(sequences_d_));
    GW_CU_CHECK_ERR(cudaFree(results_d_));
    GW_CU_CHECK_ERR(cudaFree(sequence_lengths_d_));
    GW_CU_CHECK_ERR(cudaFree(result_lengths_d_));

    GW_CU_CHECK_ERR(cudaFreeHost(sequences_h_));
    GW_CU_CHECK_ERR(cudaFreeHost(sequence_lengths_h_));
    GW_CU_CHECK_ERR(cudaFreeHost(results_h_));
    GW_CU_CHECK_ERR(cudaFreeHost(result_lengths_h_));
}

StatusType AlignerGlobal::add_alignment(const char* query, uint32_t query_length, const char* subject, uint32_t subject_length)
{
    uint32_t const max_alignment_length           = std::max(max_query_length_, max_subject_length_);
    int32_t const allocated_max_length_difference = static_cast<int32_t>(max_subject_length_ * max_target_query_length_difference);
    uint32_t const num_alignments                 = alignments_.size();
    if (num_alignments >= max_alignments_)
    {
        GW_LOG_DEBUG("{} {}", "Exceeded maximum number of alignments allowed : ", max_alignments_);
        return StatusType::exceeded_max_alignments;
    }

    if (query_length > max_query_length_)
    {
        GW_LOG_DEBUG("{} {}", "Exceeded maximum length of query allowed : ", max_query_length_);
        return StatusType::exceeded_max_length;
    }

    if (subject_length > max_subject_length_)
    {
        GW_LOG_DEBUG("{} {}", "Exceeded maximum length of subject allowed : ", max_subject_length_);
        return StatusType::exceeded_max_length;
    }

    if (std::abs(static_cast<int32_t>(query_length) - static_cast<int32_t>(subject_length)) > allocated_max_length_difference)
    {
        GW_LOG_DEBUG("{} {}", "Exceeded maximum length difference b/w subject and query allowed : ", allocated_max_length_difference);
        return StatusType::exceeded_max_alignment_difference;
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
    int32_t const max_alignment_length            = std::max(max_query_length_, max_subject_length_);
    int32_t const num_alignments                  = alignments_.size();
    int32_t const allocated_max_length_difference = static_cast<int32_t>(max_subject_length_ * max_target_query_length_difference);
    int32_t const ukkonen_p                       = 100;
    if (!score_matrices_)
        score_matrices_ = std::make_unique<batched_device_matrices<nw_score_t>>(
            max_alignments_, ukkonen_max_score_matrix_size(max_query_length_, max_subject_length_, allocated_max_length_difference, ukkonen_p), stream_, device_id_);
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(sequence_lengths_d_,
                                    sequence_lengths_h_,
                                    2 * sizeof(uint32_t) * num_alignments,
                                    cudaMemcpyHostToDevice,
                                    stream_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(sequences_d_,
                                    sequences_h_,
                                    2 * sizeof(char) * max_alignment_length * num_alignments,
                                    cudaMemcpyHostToDevice,
                                    stream_));

    int32_t max_length_difference = 0;
    for (int32_t i = 0; i < num_alignments; ++i)
    {
        max_length_difference = std::max(max_length_difference,
                                         std::abs(sequence_lengths_h_[2 * i] - sequence_lengths_h_[2 * i + 1]));
    }

    // Run kernel
    ukkonen_gpu(
        results_d_, result_lengths_d_, max_query_length_ + max_subject_length_,
        sequences_d_, sequence_lengths_d_,
        max_length_difference, max_alignment_length, num_alignments,
        score_matrices_.get(),
        ukkonen_p,
        stream_);

    GW_CU_CHECK_ERR(cudaMemcpyAsync(results_h_,
                                    results_d_,
                                    sizeof(uint8_t) * (max_query_length_ + max_subject_length_) * num_alignments,
                                    cudaMemcpyDeviceToHost,
                                    stream_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(result_lengths_h_,
                                    result_lengths_d_,
                                    sizeof(uint32_t) * num_alignments,
                                    cudaMemcpyDeviceToHost,
                                    stream_));
    return StatusType::success;
}

StatusType AlignerGlobal::sync_alignments()
{
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    GW_CU_CHECK_ERR(cudaStreamSynchronize(stream_));

    int32_t const n_alignments = alignments_.size();
    std::vector<AlignmentState> al_state;
    for (int32_t i = 0; i < n_alignments; ++i)
    {
        al_state.clear();
        int8_t const* r_begin = results_h_ + i * (max_query_length_ + max_subject_length_);
        int8_t const* r_end   = r_begin + result_lengths_h_[i];
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
}
}
