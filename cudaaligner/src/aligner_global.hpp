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

#include "ukkonen_gpu.cuh"

#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>
#include <claraparabricks/genomeworks/utils/allocator.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>
#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

class AlignerGlobal : public Aligner
{
public:
    AlignerGlobal(int32_t max_query_length, int32_t max_target_length, int32_t max_alignments, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id);
    virtual ~AlignerGlobal()            = default;
    AlignerGlobal(const AlignerGlobal&) = delete;

    virtual StatusType align_all() override;

    virtual StatusType sync_alignments() override;

    virtual StatusType add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target) override;

    virtual const std::vector<std::shared_ptr<Alignment>>& get_alignments() const override
    {
        return alignments_;
    }

    virtual int32_t num_alignments() const
    {
        return get_size(alignments_);
    }

    virtual void reset() override;

    int32_t get_max_target_length() const
    {
        return max_target_length_;
    }

    int32_t get_max_query_length() const
    {
        return max_query_length_;
    }

private:
    virtual void run_alignment(int8_t* results_d, int32_t* result_lengths, int32_t max_result_length, const char* sequences_d, int32_t* sequence_lengths_d, int32_t* sequence_lengths_h, int32_t max_sequence_length, int32_t num_alignments, cudaStream_t stream) = 0;

    int32_t max_query_length_;
    int32_t max_target_length_;
    int32_t max_alignments_;
    std::vector<std::shared_ptr<Alignment>> alignments_;

    device_buffer<char> sequences_d_;
    pinned_host_vector<char> sequences_h_;

    device_buffer<int32_t> sequence_lengths_d_;
    pinned_host_vector<int32_t> sequence_lengths_h_;

    device_buffer<int8_t> results_d_;
    pinned_host_vector<int8_t> results_h_;

    device_buffer<int32_t> result_lengths_d_;
    pinned_host_vector<int32_t> result_lengths_h_;

    cudaStream_t stream_;
    int32_t device_id_;
};

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
