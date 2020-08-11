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

#include "overlapper_anchmer.hpp"

#include <fstream>
#include <cstdlib>

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>

#include <claraparabricks/genomeworks/utils/cudautils.hpp>

#ifndef NDEBUG // only needed to check if input is sorted in assert
#include <algorithm>
#include <thrust/host_vector.h>
#endif

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

struct Anchmer
{
    std::int32_t n_anchors = 0;
    std::int32_t n_chained_anchors [10] = {0};
    std::int32_t chain_id [10] = {0};
    
};

__device__ bool operator==(const Anchor& lhs,
                                    const Anchor& rhs)
{
    auto score_threshold = 1;

    // Very simple scoring function to quantify quality of overlaps.
    auto score = 1;

    if ((rhs.query_position_in_read_ - lhs.query_position_in_read_) < 150 and
     abs(int(rhs.target_position_in_read_) - int(lhs.target_position_in_read_)) < 150)
        score = 2;
    return ((lhs.query_read_id_ == rhs.query_read_id_) &&
            (lhs.target_read_id_ == rhs.target_read_id_) &&
            score > score_threshold);
}


__global__  void
generate_anchmers(const Anchor* d_anchors, const size_t n_anchors, Anchmer* anchmers, const uint8_t anchmer_size)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;

    std::size_t first_anchor_index = d_tid * anchmer_size;

    anchmers[d_tid].n_anchors = 0;
    std::int32_t current_chain = 1;
    for (int i = 0; i < 10; ++i){
        anchmers[d_tid].chain_id[i] = 0;
    }
    anchmers[d_tid].chain_id[0] = current_chain;

    for (std::size_t i = 0; i < anchmer_size; ++i){
        std::size_t global_anchor_index = first_anchor_index + i;
        if (global_anchor_index < n_anchors){
            ++(anchmers[d_tid].n_anchors);
            anchmers[d_tid].chain_id[i] = anchmers[d_tid].chain_id[i] == 0 ? ++current_chain : anchmers[d_tid].chain_id[i];
            std::size_t j = i + 1;
            while(j < anchmer_size && j + first_anchor_index < n_anchors)
            {
                if (d_anchors[global_anchor_index] == d_anchors[first_anchor_index + j]){
                    anchmers[d_tid].chain_id[j] = anchmers[d_tid].chain_id[i];
                }
                ++j;
            }
        }
    }


}

void OverlapperAnchmer::get_overlaps(std::vector<Overlap>& fused_overlaps,
                                     const device_buffer<Anchor>& d_anchors,
                                     bool all_to_all,
                                     int64_t min_residues,
                                     int64_t min_overlap_len,
                                     int64_t min_bases_per_residue,
                                     float min_overlap_fraction)
{

    // const std::int32_t anchmer_generation_rounds = 1;
    // const std::int32_t chain_filter_min_anchors  = 2;
    // const std::int32_t anchor_merge_min_dist     = 150;
    const std::int32_t anchors_per_anchmer = 10;
    std::size_t n_anchors = d_anchors.size();
    std::size_t n_anchmers = (d_anchors.size() / anchors_per_anchmer) + 1; 
    std::int32_t block_size = 32;

    std::vector<Anchmer> anchmers(n_anchmers);
    device_buffer<Anchmer> d_anchmers(n_anchmers, _allocator, _cuda_stream);

    // Stage one: generate anchmers
    generate_anchmers<<<(n_anchmers / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchors.data(), n_anchors, d_anchmers.data(), anchors_per_anchmer);

    cudautils::device_copy_n(d_anchmers.data(), d_anchmers.size(), anchmers.data());

    for (auto a : anchmers){
        std::cout << a.n_anchors << std::endl;
        for (std::size_t i = 0; i < a.n_anchors; ++i){
            std::cout << a.chain_id[i] << " ";
        }
        std::cout << std::endl;
    }

    // Stage two: within-anchmer chaining

    // Stage 3 (optional): within-anchmer chain filtering

    // Stage 4: anchmer superchaining (merge anchmers which share chains)
}

OverlapperAnchmer::OverlapperAnchmer(DefaultDeviceAllocator allocator,
                                     const cudaStream_t cuda_stream)
    : _allocator(allocator)
    , _cuda_stream(cuda_stream)
{
}

} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks
