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

#include <cuda_runtime_api.h>
#include "batched_device_matrices.cuh"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

namespace hirschbergmyers
{
using WordType   = uint32_t;
using nw_score_t = int32_t;

struct query_target_range
{
    char const* query_begin  = nullptr;
    char const* query_end    = nullptr;
    char const* target_begin = nullptr;
    char const* target_end   = nullptr;
};
} // namespace hirschbergmyers

void hirschberg_myers_gpu(device_buffer<hirschbergmyers::query_target_range>& stack_buffer,
                          int32_t stacksize_per_alignment,
                          int8_t* paths_d, int32_t* path_lengths_d, int32_t max_path_length,
                          char const* sequences_d,
                          int32_t const* sequence_lengths_d,
                          int32_t max_target_query_length,
                          int32_t n_alignments,
                          batched_device_matrices<hirschbergmyers::WordType>& pv,
                          batched_device_matrices<hirschbergmyers::WordType>& mv,
                          batched_device_matrices<int32_t>& score,
                          batched_device_matrices<hirschbergmyers::WordType>& query_patterns,
                          int32_t switch_to_myers_threshold,
                          cudaStream_t stream);

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
