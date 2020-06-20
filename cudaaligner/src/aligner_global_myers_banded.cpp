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

#include "aligner_global_myers_banded.hpp"
#include "myers_gpu.cuh"
#include "batched_device_matrices.cuh"

#include <claraparabricks/genomeworks/utils/mathutils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

namespace
{

constexpr int32_t word_size                = sizeof(myers::WordType) * CHAR_BIT;

int64_t compute_matrix_size_per_alignment(int32_t max_target_length, int32_t max_bandwidth)
{
    assert(max_bandwidth >= 0);
    assert(max_target_length >= 0);
    const int32_t query_size            = max_target_length;
    const int32_t p                     = (max_bandwidth+1) / 2;
    const int32_t band_width            = std::min(1 + 2 * p, query_size);
    const int64_t n_words_band          = ceiling_divide(band_width, word_size);
    return n_words_band * (static_cast<int64_t>(max_target_length) + 1);
}

} // namespace

int64_t AlignerGlobalMyersBanded::calc_memory_requirement_per_alignment(int32_t max_sequence_length, int32_t max_bandwidth)
{
    constexpr int32_t alignment_bytes = 4;
    const int32_t max_query_length = max_sequence_length;
    const int32_t max_target_length = max_sequence_length;
    const int32_t max_words_query = ceiling_divide(max_query_length, word_size);
    const int64_t matrix_size_per_alignment = compute_matrix_size_per_alignment(max_target_length, max_bandwidth);
    const int32_t max_result_length = ceiling_divide(max_query_length + max_target_length, alignment_bytes) * alignment_bytes;
    return 2 * matrix_size_per_alignment * sizeof(myers::WordType) +
        matrix_size_per_alignment * sizeof(int32_t) +
        max_words_query * 4 * sizeof(myers::WordType) +
        2 * max_result_length * sizeof(int8_t);
}

struct AlignerGlobalMyersBanded::Workspace
{
    Workspace(int32_t max_alignments, int32_t max_n_words, int64_t matrix_size_per_alignment, DefaultDeviceAllocator allocator, cudaStream_t stream)
        : pvs(max_alignments, matrix_size_per_alignment, allocator, stream)
        , mvs(max_alignments, matrix_size_per_alignment, allocator, stream)
        , scores(max_alignments, matrix_size_per_alignment, allocator, stream)
        , query_patterns(max_alignments, max_n_words * 4, allocator, stream)
    {
    }
    batched_device_matrices<myers::WordType> pvs;
    batched_device_matrices<myers::WordType> mvs;
    batched_device_matrices<int32_t> scores;
    batched_device_matrices<myers::WordType> query_patterns;
};

AlignerGlobalMyersBanded::AlignerGlobalMyersBanded(int32_t max_sequence_length, int32_t max_bandwidth, int32_t max_alignments, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
    : AlignerGlobal(max_sequence_length, max_sequence_length, max_alignments, allocator, stream, device_id)
    , workspace_()
    , max_bandwidth_(max_bandwidth)
{
    if(max_bandwidth % (sizeof(myers::WordType)*CHAR_BIT) == 1)
    {
        throw std::runtime_error("Invalid max_bandwidth value. Please change it by +/-1.");
    }
    scoped_device_switch dev(device_id);
    const int32_t max_words_query           = ceiling_divide<int32_t>(max_sequence_length, word_size);
    const int64_t matrix_size_per_alignment = compute_matrix_size_per_alignment(max_sequence_length, max_bandwidth_);
    workspace_ = std::make_unique<Workspace>(max_alignments, max_words_query, matrix_size_per_alignment, allocator, stream);
}

AlignerGlobalMyersBanded::~AlignerGlobalMyersBanded()
{
    // Keep empty destructor to keep Workspace type incomplete in the .hpp file.
}

void AlignerGlobalMyersBanded::run_alignment(int8_t* results_d, int32_t* result_lengths_d, int32_t max_result_length,
                                             const char* sequences_d, int32_t* sequence_lengths_d, int32_t* sequence_lengths_h, int32_t max_sequence_length,
                                             int32_t num_alignments, cudaStream_t stream)
{
    static_cast<void>(sequence_lengths_h);
    myers_banded_gpu(results_d, result_lengths_d, max_result_length,
                     sequences_d, sequence_lengths_d, max_sequence_length, num_alignments, max_bandwidth_,
                     workspace_->pvs, workspace_->mvs, workspace_->scores, workspace_->query_patterns,
                     stream);
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
