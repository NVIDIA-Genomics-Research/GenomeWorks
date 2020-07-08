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

#include "aligner_global.hpp"
#include "ukkonen_gpu.cuh"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

template <typename T>
class batched_device_matrices;

class AlignerGlobalUkkonen : public AlignerGlobal
{
public:
    AlignerGlobalUkkonen(int32_t max_query_length, int32_t max_target_length, int32_t max_alignments, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id);
    virtual ~AlignerGlobalUkkonen();
    StatusType add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target) override;

private:
    using BaseType = AlignerGlobal;

    virtual void run_alignment(int8_t* results_d, int32_t* result_lengths_d, int32_t max_result_length,
                               const char* sequences_d, int32_t* sequence_lengths_d, int32_t* sequence_lengths_h, int32_t max_sequence_length,
                               int32_t num_alignments, cudaStream_t stream) override;

    std::unique_ptr<batched_device_matrices<nw_score_t>> score_matrices_;
    int32_t ukkonen_p_;
};

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
