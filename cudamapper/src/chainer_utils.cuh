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

#include <fstream>
#include <sstream>
#include <cstdlib>

// Needed for accumulate - remove when ported to cuda
#include <numeric>
#include <limits>

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

#define MAX_CHAINS_PER_TILE 5

struct QueryTargetPair
{
    int32_t query_read_id_;
    int32_t target_read_id_;
    __device__ QueryTargetPair() {}
};

struct QueryReadID
{
    int32_t query_read_id_;
    __device__ QueryReadID(){};
};

// takes the anchor and returns the query read id
struct AnchorToQueryReadIDOp
{
    __device__ __forceinline__ QueryReadID operator()(const Anchor& a) const
    {
        QueryReadID query;
        query.query_read_id_ = a.query_read_id_;
        return query;
    }
};

__device__ bool operator==(const QueryTargetPair& a, const QueryTargetPair& b);

struct OverlapToQueryTargetPairOp
{
    __device__ __forceinline__ QueryTargetPair operator()(const Overlap& a) const
    {
        QueryTargetPair p;
        p.query_read_id_  = a.query_read_id_;
        p.target_read_id_ = a.target_read_id_;
        return p;
    }
};

struct AnchorToQueryTargetPairOp
{
    __device__ __forceinline__ QueryTargetPair operator()(const Anchor& a) const
    {
        QueryTargetPair p;
        p.query_read_id_  = a.query_read_id_;
        p.target_read_id_ = a.target_read_id_;
        return p;
    }
};

struct OverlapToNumResiduesOp
{
    __device__ __forceinline__ int32_t operator()(const Overlap& overlap) const
    {
        return overlap.num_residues_;
    }
};

__device__ bool
operator==(const QueryTargetPair& a, const QueryTargetPair& b);

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
void allocate_anchor_chains(device_buffer<Overlap> overlaps,
                            device_buffer<int32_t>& unrolled_anchor_chains,
                            const int32_t num_overlaps,
                            int32_t& num_total_anchors,
                            DefaultDeviceAllocator& _allocator,
                            cudaStream_t& _cuda_stream);

__global__ void output_overlap_chains(const Overlap* overlaps,
                                      const Anchor* anchors,
                                      const bool* select_mask,
                                      const int32_t* predecessors,
                                      int32_t* anchor_chains,
                                      int32_t* anchor_chain_starts,
                                      int32_t num_overlaps,
                                      bool check_mask);

__global__ void convert_offsets_to_ends(std::int32_t* starts, std::int32_t* lengths, std::int32_t* ends, std::int32_t n_starts);

__global__ void calculate_tile_starts(const std::int32_t* query_starts,
                                      const std::int32_t* tiles_per_query,
                                      std::int32_t* tile_starts,
                                      const int32_t tile_size,
                                      int32_t num_queries,
                                      const std::int32_t* tiles_per_query_up_to_point);

void encode_anchor_query_locations(const Anchor* anchors,
                                   int32_t n_anchors,
                                   int32_t tile_size,
                                   device_buffer<int32_t>& query_starts,
                                   device_buffer<int32_t>& query_lengths,
                                   device_buffer<int32_t>& query_ends,
                                   device_buffer<int32_t>& tiles_per_query,
                                   device_buffer<int32_t>& tile_starts,
                                   int32_t& n_queries,
                                   int32_t& n_query_tiles,
                                   DefaultDeviceAllocator& _allocator,
                                   cudaStream_t& _cuda_stream,
                                   int32_t block_size);

void encode_anchor_query_target_pairs(const Anchor* anchors,
                                      int32_t n_anchors,
                                      int32_t tile_size,
                                      device_buffer<int32_t>& query_target_pair_starts,
                                      device_buffer<int32_t>& query_target_pair_lengths,
                                      device_buffer<int32_t>& query_target_pair_ends,
                                      device_buffer<int32_t>& tiles_per_qt_pair,
                                      int32_t& n_query_target_pairs,
                                      int32_t& n_qt_tiles,
                                      DefaultDeviceAllocator& _allocator,
                                      cudaStream_t& _cuda_stream,
                                      int32_t block_size);

void encode_overlap_query_target_pairs(Overlap* overlaps,
                                       int32_t n_overlaps,
                                       device_buffer<int32_t>& query_target_pair_starts,
                                       device_buffer<int32_t>& query_target_pair_lengths,
                                       device_buffer<int32_t>& query_target_pair_ends,
                                       int32_t& n_query_target_pairs,
                                       DefaultDeviceAllocator& _allocator,
                                       cudaStream_t& _cuda_stream,
                                       int32_t block_size);
} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks