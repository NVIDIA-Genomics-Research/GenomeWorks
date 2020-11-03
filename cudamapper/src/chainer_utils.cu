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

#include "chainer_utils.cuh"

#include <cub/cub.cuh>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>

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

__host__ __device__ Overlap create_overlap(const Anchor& start, const Anchor& end, const int32_t num_anchors)
{
    Overlap overlap;
    overlap.num_residues_ = num_anchors;

    overlap.query_read_id_  = start.query_read_id_;
    overlap.target_read_id_ = start.target_read_id_;
    assert(start.query_read_id_ == end.query_read_id_ && start.target_read_id_ == end.target_read_id_);

    overlap.query_start_position_in_read_ = min(start.query_position_in_read_, end.query_position_in_read_);
    overlap.query_end_position_in_read_   = max(start.query_position_in_read_, end.query_position_in_read_);
    const bool is_negative_strand         = end.target_position_in_read_ < start.target_position_in_read_;
    if (is_negative_strand)
    {
        overlap.relative_strand                = RelativeStrand::Reverse;
        overlap.target_start_position_in_read_ = end.target_position_in_read_;
        overlap.target_end_position_in_read_   = start.target_position_in_read_;
    }
    else
    {
        overlap.relative_strand                = RelativeStrand::Forward;
        overlap.target_start_position_in_read_ = start.target_position_in_read_;
        overlap.target_end_position_in_read_   = end.target_position_in_read_;
    }
    return overlap;
}

__global__ void backtrace_anchors_to_overlaps(const Anchor* const anchors,
                                              Overlap* overlaps,
                                              double* const scores,
                                              bool* const max_select_mask,
                                              int32_t* const predecessors,
                                              const int64_t n_anchors,
                                              const int32_t min_score)
{
    const int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n_anchors; i += stride)
    {
        if (scores[i] >= min_score)
        {
            int32_t index                = i;
            int32_t first_index          = index;
            int32_t num_anchors_in_chain = 0;
            Anchor final_anchor          = anchors[i];

            while (index != -1)
            {
                first_index  = index;
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    max_select_mask[pred] = false;
                }
                num_anchors_in_chain++;
                index = predecessors[index];
            }
            Anchor first_anchor = anchors[first_index];
            overlaps[i]         = create_overlap(first_anchor, final_anchor, num_anchors_in_chain);
        }
        else
        {
            max_select_mask[i] = false;
        }
    }
}

void allocate_anchor_chains(const device_buffer<Overlap>& overlaps,
                            device_buffer<int32_t>& unrolled_anchor_chains,
                            device_buffer<int32_t>& anchor_chain_starts,
                            int64_t& num_total_anchors,
                            DefaultDeviceAllocator allocator,
                            cudaStream_t cuda_stream)
{
    // sum the number of chains across all overlaps

    void* temp_storage_d           = nullptr;
    std::size_t temp_storage_bytes = 0;
    OverlapToNumResiduesOp overlap_residue_count_op;
    cub::TransformInputIterator<int32_t, OverlapToNumResiduesOp, const Overlap*> residue_counts_d(overlaps.data(),
                                                                                                  overlap_residue_count_op);

    device_buffer<int64_t> num_total_anchors_d(1, allocator, cuda_stream);

    cub::DeviceReduce::Sum(temp_storage_d,
                           temp_storage_bytes,
                           residue_counts_d,
                           num_total_anchors_d.data(),
                           overlaps.size(),
                           cuda_stream);

    device_buffer<char> temp_buf_d(temp_storage_bytes, allocator, cuda_stream);
    temp_storage_d = temp_buf_d.data();

    cub::DeviceReduce::Sum(temp_storage_d,
                           temp_storage_bytes,
                           residue_counts_d,
                           num_total_anchors_d.data(),
                           overlaps.size(),
                           cuda_stream);

    temp_storage_d     = nullptr;
    temp_storage_bytes = 0;

    num_total_anchors = cudautils::get_value_from_device(num_total_anchors_d.data(), cuda_stream);

    unrolled_anchor_chains.clear_and_resize(num_total_anchors);
    anchor_chain_starts.clear_and_resize(overlaps.size());

    cub::DeviceScan::ExclusiveSum(temp_storage_d,
                                  temp_storage_bytes,
                                  residue_counts_d,
                                  anchor_chain_starts.data(),
                                  overlaps.size(),
                                  cuda_stream);

    temp_buf_d.clear_and_resize(temp_storage_bytes);
    temp_storage_d = temp_buf_d.data();

    cub::DeviceScan::ExclusiveSum(temp_storage_d,
                                  temp_storage_bytes,
                                  residue_counts_d,
                                  anchor_chain_starts.data(),
                                  overlaps.size(),
                                  cuda_stream);
}

__global__ void output_overlap_chains_by_backtrace(const Overlap* const overlaps,
                                                   const Anchor* const anchors,
                                                   const bool* const select_mask,
                                                   const int32_t* const predecessors,
                                                   int32_t* const anchor_chains,
                                                   int32_t* const anchor_chain_starts,
                                                   const int32_t num_overlaps,
                                                   const bool check_mask)
{
    const int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride    = blockDim.x * gridDim.x;

    // Processes one overlap per iteration,
    // "i" corresponds to an overlap
    for (int i = thread_id; i < num_overlaps; i += stride)
    {
        // index within this chain of anchors (i.e., the anchors within a single overlap)

        if (!check_mask || (check_mask & select_mask[i]))
        {
            int32_t anchor_chain_index = 0;
            // As chaining proceeds backwards (i.e., it's a backtrace),
            // we need to fill the new anchor chain array in in reverse order.
            int32_t index = anchor_chain_starts[i];
            while (index != -1)
            {
                anchor_chains[anchor_chain_starts[i] + (overlaps[i].num_residues_ - anchor_chain_index)] = index;
                index                                                                                    = predecessors[index];
                ++anchor_chain_index;
            }
        }
    }
}

__global__ void output_overlap_chains_by_RLE(const Overlap* const overlaps,
                                             const Anchor* const anchors,
                                             const int32_t* const chain_starts,
                                             const int32_t* const chain_lengths,
                                             int32_t* const anchor_chains,
                                             int32_t* const anchor_chain_starts,
                                             const uint32_t num_overlaps)
{
    const int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t stride    = blockDim.x * gridDim.x;
    for (uint32_t i = thread_id; i < num_overlaps; i += stride)
    {
        int32_t chain_start  = chain_starts[i];
        int32_t chain_length = chain_lengths[i];
        for (int32_t ind = chain_start; ind < chain_start + chain_length; ++i)
        {
            anchor_chains[ind] = ind;
        }
    }
}

} // namespace chainerutils
} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks