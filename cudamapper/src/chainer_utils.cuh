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

///
/// \brief Create an overlap from the first and last anchor in the chain and
/// the total number of anchors in the chain.
///
/// \param start The first anchor in the chain.
/// \param end The last anchor in the chain.
/// \param num_anchors The total number of anchors in the chain.
__host__ __device__ Overlap create_simple_overlap(const Anchor& start,
                                                  const Anchor& end,
                                                  const int32_t num_anchors);

///
/// \brief Perform a backtrace on the predecessors array
/// to generate overlaps from anchors. Anchors must have been chained
/// by a chaining function that filles the predecessors and scores array.
///
/// \param anchors An array of anchors.
/// \param overlaps An array of overlaps to be filled.
/// \param scores An array of scores. Only chains with a score greater than
/// min_score will be backtraced.
/// \param max_select_mask A boolean mask, used to mask out any subchains during
/// the backtrace.
/// \param predecessors An array of indices into the anchors array marking the predecessor
/// of each anchor within a chain.
/// \param n_anchors The number of anchors.
/// \param min_score The minimum score of a chain for performing backtracing.
/// \return __global__
__global__ void backtrace_anchors_to_overlaps(const Anchor* anchors,
                                              Overlap* overlaps,
                                              double* scores,
                                              bool* max_select_mask,
                                              int32_t* predecessors,
                                              const int32_t n_anchors,
                                              const int32_t min_score);
///
/// \brief Allocate a 1-dimensional array representing an unrolled 2D-array
/// (overlap X n_anchors_in_overlap) of anchors within each overlap. Rather than
/// copy the anchors, the final array holds the indices within the anchors array
/// of the anchors in the chain.
///
/// \param overlaps An array of Overlaps. Must have a well-formed num_residues_ field.
/// \param unrolled_anchor_chains  An array of int32_t. Will be resided on return.
/// param num_overlaps The number of overlaps in the overlaps array.
///@param _allocator  The DefaultDeviceAllocator
///@param _cuda_stream The cudastream to allocate memory within.

///
/// \brief Allocate a 1-dimensional array representing an unrolled 2D-array
/// (overlap X n_anchors_in_overlap) of anchors within each overlap. Rather than
/// copy the anchors, the final array holds the indices within the anchors array
/// of the anchors in the chain.
///
/// \param overlaps An array of Overlaps. Must have a well-formed num_residues_ field
/// \param unrolled_anchor_chains  An array of int32_t. Will be resided on return.
/// \param anchor_chain_starts An array holding the index in the anchors array of the first
/// anchor in an overlap.
/// \param num_overlaps The number of overlaps in the overlaps array.
/// \param num_total_anchors The number of anchors in the anchors array.
/// \param _allocator The DefaultDeviceAllocator for this overlapper.
/// \param _cuda_stream The cudastream to allocate memory within.
void allocate_anchor_chains(device_buffer<Overlap>& overlaps,
                            device_buffer<int32_t>& unrolled_anchor_chains,
                            device_buffer<int32_t>& anchor_chain_starts,
                            int32_t num_overlaps,
                            int32_t& num_total_anchors,
                            DefaultDeviceAllocator& _allocator,
                            cudaStream_t& _cuda_stream);

///
/// \brief Calculate the anchors chains used to produce each overlap in the
/// overlap array for anchors chained by backtrace_anchors_to_overlaps.
///
/// \param overlaps An array of overlaps.
/// \param anchors The array of anchors used to generate overlaps.
/// \param select_mask An array of bools, used to mask overlaps from output.
/// \param predecessors The predecessors array from anchor chaining.
/// \param anchor_chains An array (allocated by allocate_anchor_chains) which will hold
/// the indices of anchors within each chain.
/// \param anchor_chain_starts An array which holds the indices of the first anchor for
/// each overlap in the overlaps array.
/// \param num_overlaps The number of overlaps in the overlaps array
/// \param check_mask A boolean. If true, only overlaps where select_mask is true
/// will have their anchor chains calculated.
/// \return __global__
__global__ void output_overlap_chains_by_backtrace(const Overlap* overlaps,
                                                   const Anchor* anchors,
                                                   const bool* select_mask,
                                                   const int32_t* predecessors,
                                                   int32_t* anchor_chains,
                                                   int32_t* anchor_chain_starts,
                                                   int32_t num_overlaps,
                                                   bool check_mask);
///
/// \brief Calculate the anchors chains used to produce each overlap in the
/// overlap array for anchors chained by RLE.
///
/// \param overlaps An array of overlaps.
/// \param anchors The array of anchors used to generate overlaps.
/// \param chain_starts An array which holds the indices of the first anchor for
/// each overlap in the overlaps array.
/// \param chain_lengths An array which holds the length of each run of anchors, corresponding to
/// the chain_starts array.
/// \param anchor_chains An array (allocated by allocate_anchor_chains) which will hold
/// the indices of anchors within each chain.
/// \param anchor_chain_starts An array which holds the indices of the first anchor for
/// each overlap in the overlaps array.
/// \param num_overlaps The number of overlaps in the overlaps array.
/// \return __global__
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