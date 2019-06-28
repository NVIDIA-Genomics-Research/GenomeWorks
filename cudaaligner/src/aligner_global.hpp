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

#include "cudaaligner/aligner.hpp"
#include "ukkonen_gpu.cuh"
#include "device_storage.cuh"
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace genomeworks
{

namespace cudaaligner
{

template <typename T>
class batched_device_matrices;

class AlignerGlobal : public Aligner
{
public:
    AlignerGlobal(int32_t max_query_length, int32_t max_subject_length, int32_t max_alignments, int32_t device_id);
    virtual ~AlignerGlobal();
    AlignerGlobal(const AlignerGlobal&) = delete;

    virtual StatusType align_all() override;

    virtual StatusType sync_alignments() override;

    virtual StatusType add_alignment(const char* query, int32_t query_length, const char* subject, int32_t subject_length) override;

    virtual const std::vector<std::shared_ptr<Alignment>>& get_alignments() const override
    {
        return alignments_;
    }

    virtual int32_t num_alignments() const
    {
        return alignments_.size();
    }

    virtual void set_cuda_stream(cudaStream_t stream) override
    {
        stream_ = stream;
    }

    virtual void reset() override;

private:
    template <typename T>
    using pinned_host_vector = std::vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;

    int32_t max_query_length_;
    int32_t max_subject_length_;
    int32_t max_alignments_;
    std::vector<std::shared_ptr<Alignment>> alignments_;

    device_storage<char> sequences_d_;
    pinned_host_vector<char> sequences_h_;

    device_storage<int32_t> sequence_lengths_d_;
    pinned_host_vector<int32_t> sequence_lengths_h_;

    device_storage<int8_t> results_d_;
    pinned_host_vector<int8_t> results_h_;

    device_storage<int32_t> result_lengths_d_;
    pinned_host_vector<int32_t> result_lengths_h_;

    std::unique_ptr<batched_device_matrices<nw_score_t>> score_matrices_;

    cudaStream_t stream_;

    int32_t device_id_;
};
} // namespace cudaaligner
} // namespace genomeworks
