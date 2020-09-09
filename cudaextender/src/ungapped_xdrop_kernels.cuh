/*
* Copyright 2020 NVIDIA CORPORATION.
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

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include "ungapped_xdrop.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

// extend the hits to a segment by ungapped x-drop method, adjust low-scoring
// segment scores based on entropy factor, compare resulting segment scores
// to hspthresh and update the d_hsp and d_done vectors
__global__ void find_high_scoring_segment_pairs(const char* __restrict__ d_target, const int32_t target_length, const char* __restrict__ d_query, const int32_t query_length, const int* d_sub_mat, bool no_entropy, int32_t xdrop_threshold, int32_t score_threshold, SeedPair* d_seed_pairs, int32_t num_seed_pairs, int32_t start_index, ScoredSegmentPair* d_scored_segment, uint32_t* d_done);


}

}

}
