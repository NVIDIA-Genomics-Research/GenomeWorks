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

#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
namespace chainerutils
{

struct OverlapToNumResiduesOp
{
    __device__ __forceinline__ int32_t operator()(const Overlap& overlap) const
    {
        return overlap.num_residues_;
    }
};

__host__ __device__ Overlap create_simple_overlap(const Anchor& start,
                                                  const Anchor& end,
                                                  const int32_t num_anchors);

__global__ void backtrace_anchors_to_overlaps(const Anchor* anchors,
                                              Overlap* overlaps,
                                              double* scores,
                                              bool* max_select_mask,
                                              int32_t* predecessors,
                                              const int32_t n_anchors,
                                              const int32_t min_score);
///
///@brief Allocate a 1-dimensional array representing an unrolled 2D-array
/// (overlap X n_anchors_in_overlap) of anchors within each overlap. Rather than
/// copy the anchors, the final array holds the indices of the global index array.
///
///@param overlaps An array of Overlaps. Must have a well-formed num_residues_ field.
///@param unrolled_anchor_chains  An array of int32_t. Will be resided on return.
///@param num_overlaps The number of overlaps in the overlaps array.
///@param _allocator  The DefaultDeviceAllocator
///@param _cuda_stream The cudastream to allocate memory within.
void allocate_anchor_chains(device_buffer<Overlap>& overlaps,
                            device_buffer<int32_t>& unrolled_anchor_chains,
                            device_buffer<int32_t>& anchor_chain_starts,
                            int32_t num_overlaps,
                            int32_t& num_total_anchors,
                            DefaultDeviceAllocator& _allocator,
                            cudaStream_t& _cuda_stream);

__global__ void output_overlap_chains_by_backtrace(const Overlap* overlaps,
                                                   const Anchor* anchors,
                                                   const bool* select_mask,
                                                   const int32_t* predecessors,
                                                   int32_t* anchor_chains,
                                                   int32_t* anchor_chain_starts,
                                                   int32_t num_overlaps,
                                                   bool check_mask);

__global__ void output_overlap_chains_by_RLE(const Overlap* overlaps,
                                             const Anchor* anchors,
                                             const int32_t* chain_starts,
                                             const int32_t* chain_lengths,
                                             int32_t* anchor_chains,
                                             int32_t* anchor_chain_starts,
                                             int32_t num_overlaps);

} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks