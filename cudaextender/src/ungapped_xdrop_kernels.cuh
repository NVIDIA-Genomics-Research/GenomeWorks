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

/ extend the hits to a segment by ungapped x-drop method, adjust low-scoring
// segment scores based on entropy factor, compare resulting segment scores
// to hspthresh and update the d_hsp and d_done vectors
__global__ void find_hsps (const char* __restrict__  d_ref_seq, const char* __restrict__  d_query_seq, uint32_t ref_len, uint32_t query_len, int *d_sub_mat, bool noentropy, int xdrop, int hspthresh, int num_hits, seedHit* d_hit, uint32_t start_index, segment* d_hsp, uint32_t* d_done);
