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

__device__ __forceinline__ Anchor empty_anchor()
{
    Anchor empty;
    empty.query_read_id_           = UINT32_MAX;
    empty.target_read_id_          = UINT32_MAX;
    empty.query_position_in_read_  = UINT32_MAX;
    empty.target_position_in_read_ = UINT32_MAX;
    return empty;
}

__device__ bool
operator==(const QueryTargetPair& a, const QueryTargetPair& b);

__global__ void backtrace_anchors_to_overlaps(const Anchor* anchors,
                                              Overlap* overlaps,
                                              double* scores,
                                              bool* max_select_mask,
                                              int32_t* predecessors,
                                              const int32_t n_anchors,
                                              const int32_t min_score);

__global__ void backtrace_anchors_to_overlaps_debug(const Anchor* anchors,
                                                    Overlap* overlaps,
                                                    double* scores,
                                                    bool* max_select_mask,
                                                    int32_t* predecessors,
                                                    const int32_t n_anchors,
                                                    const int32_t min_score);

void backtrace_anchors_to_overlaps_cpu(const Anchor* anchors,
                                       Overlap* overlaps,
                                       double* scores,
                                       bool* max_select_mask,
                                       int32_t* predecessors,
                                       const int32_t n_anchors,
                                       const int32_t min_score);

__global__ void convert_offsets_to_ends(std::int32_t* starts, std::int32_t* lengths, std::int32_t* ends, std::int32_t n_starts);

__global__ void calculate_tile_starts(const std::int32_t* query_starts,
                                      const std::int32_t* tiles_per_query,
                                      std::int32_t* tile_starts,
                                      const int32_t tile_size,
                                      int32_t num_queries,
                                      const std::int32_t* tiles_per_query_up_to_point);

void encode_query_locations_from_anchors(const Anchor* anchors,
                                         int32_t n_anchors,
                                         device_buffer<int32_t>& query_starts,
                                         device_buffer<int32_t>& query_lengths,
                                         device_buffer<int32_t>& query_ends,
                                         int32_t& n_queries,
                                         DefaultDeviceAllocator& _allocator,
                                         cudaStream_t& _cuda_stream);

// void encode_tile_starts_and_ends(device_buffer<int32_t>& starts,
//                                  device_buffer<int32_t>& lengths,
//                                  device_buffer<int32_t>& ends,
//                                  device_buffer<int32_t>& tiles_per_entry,
//                                  device_buffer<int32_t>& tile_starts,
//                                  device_buffer<int32_t>& tile_ends,
//                                  int32_t num_entries,
//                                  int32_t num_tiles,
//                                  DefaultDeviceAllocator& _allocator,
//                                  cudaStream_t& _cuda_stream);

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

__global__ void initialize_mask(bool* anchors_mask, const int32_t n_anchors, bool val);

__global__ void initialize_array(int32_t* array, const int32_t num_values, int32_t value);
__global__ void initialize_array(double* array, const int32_t num_values, double value);

} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks