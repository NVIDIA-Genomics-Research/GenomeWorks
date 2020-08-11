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
    std::uint32_t n_anchors;
    Anchor* anchors;

    void merge_anchor(Anchor*& a, Anchor*& b){

    }

    void inner_chain()
    {
        std::int32_t i = 0;
        std::int32_t j  = 0;
        while (i < n_anchors)
        {
            j = i + 1;
            while (j < n_anchors)
            {
                    // if (anchors[i] == anchors[j]){
                    //     // merge_anchor(anchors[i], anchors[j]);
                    //     j = i;
                    // }
                ++j;
            }
            ++i;
        }
    }
};

__global__ __forceinline__ void
generate_anchmers(const device_buffer<Anchor>& d_anchors, const size_t n_anchors, device_buffer<Anchmer>& anchmers, const uint8_t anchmer_size)
{
    const std::uint64_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;

    const std::uint32_t n_anchmers = n_anchors / anchmer_size + 1;

    if (d_tid < n_anchors && d_tid % anchmer_size == 0)
    {
        
        for (std::size_t i = 0; i < anchmer_size; ++i)
        {

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

    const int32_t anchmer_generation_rounds = 1;
    const int32_t chain_filter_min_anchors  = 2;
    const int32_t anchor_merge_min_dist     = 150;

    std::vector<Anchmer> anchmers;
    device_buffer<Anchmer> d_anchmers;
    std::size_t n_anchors = d_anchors.size();

    generate_anchmers<<<(n_anchors + 255) / 256, 256, 0, _cuda_stream>>>(d_anchors, n_anchors, d_anchmers, 10);

    cudautils::device_copy_n(d_anchmers.data(), d_anchmers.size(), anchmers.data(), _cuda_stream);

    // Stage one: generate anchmers

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
