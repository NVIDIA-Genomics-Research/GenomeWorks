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
#include <cstdint>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

using nw_score_t = int16_t;

template <typename T>
class batched_device_matrices;

size_t ukkonen_max_score_matrix_size(int32_t max_query_length, int32_t max_target_length, int32_t max_length_difference, int32_t max_p);

void ukkonen_compute_score_matrix_gpu(batched_device_matrices<nw_score_t>& score_matrices, char const* sequences_d, int32_t const* sequence_lengths_d, int32_t max_length_difference, int32_t max_target_query_length, int32_t n_alignments, int32_t p, cudaStream_t stream);

void ukkonen_gpu(int8_t* paths_d, int32_t* path_lengths_d, int32_t max_path_length,
                 char const* sequences_d, int32_t const* sequence_lengths_d,
                 int32_t max_length_difference,
                 int32_t max_target_query_length,
                 int32_t n_alignments,
                 batched_device_matrices<nw_score_t>* score_matrices,
                 int32_t ukkonen_p,
                 cudaStream_t stream);

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
