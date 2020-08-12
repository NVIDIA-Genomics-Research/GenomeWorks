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

// Needed for accumulate - remove when ported to cuda
#include <numeric>
#include <limits>

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
    std::int8_t n_chained_anchors [10] = {0};
    std::int8_t chain_id [10] = {0};
    std::int8_t n_chains = 0;
    
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

struct SumChainsOp
{
    CUB_RUNTIME_FUNCTION __forceinline__ std::int32_t operator()(const Anchmer& a, const Anchmer& b) const
    {
        return static_cast<int32_t>(a.n_chains + a.n_chains);
    }
};

struct AnchmerCountChainsOp
{

    AnchmerCountChainsOp()
        {

        }

    __host__ __device__ __forceinline__
    std::int32_t operator()(const Anchmer& a) const {
        return static_cast<int32_t>(a.n_chains);
    }
};

struct FilterOverlapOp
{
    size_t min_residues;
    size_t min_overlap_len;
    size_t min_bases_per_residue;
    float min_overlap_fraction;
    bool indexes_identical;

    __host__ __device__ __forceinline__ FilterOverlapOp(size_t min_residues,
                                                        size_t min_overlap_len,
                                                        size_t min_bases_per_residue,
                                                        float min_overlap_fraction,
                                                        bool indexes_identical)
        : min_residues(min_residues)
        , min_overlap_len(min_overlap_len)
        , min_bases_per_residue(min_bases_per_residue)
        , min_overlap_fraction(min_overlap_fraction)
        , indexes_identical(indexes_identical)
    {
    }

    __host__ __device__ __forceinline__ bool operator()(const Overlap& overlap) const
    {

        const auto target_overlap_length = overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_;
        const auto query_overlap_length  = overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_;
        const auto overlap_length        = max(target_overlap_length, query_overlap_length);
        const bool self_mapping          = (overlap.query_read_id_ == overlap.target_read_id_) && indexes_identical;

        return ((overlap.num_residues_ >= min_residues) &&
                ((overlap_length / overlap.num_residues_) < min_bases_per_residue) &&
                (query_overlap_length >= min_overlap_len) &&
                (target_overlap_length >= min_overlap_len) &&
                (!self_mapping) &&
                ((static_cast<float>(target_overlap_length) / static_cast<float>(overlap_length)) > min_overlap_fraction) &&
                ((static_cast<float>(query_overlap_length) / static_cast<float>(overlap_length)) > min_overlap_fraction));
    }
};


__device__ __forceinline__ void add_anchor_to_overlap(const Anchor& anchor, Overlap& overlap){
    overlap.query_read_id_ = anchor.query_read_id_;
    overlap.target_read_id_ = anchor.target_read_id_;
    overlap.query_start_position_in_read_ = anchor.query_position_in_read_ < overlap.query_start_position_in_read_ ? anchor.query_position_in_read_ :  overlap.query_start_position_in_read_;
    overlap.query_end_position_in_read_ = anchor.query_position_in_read_ > overlap.query_end_position_in_read_ ? anchor.query_position_in_read_ :  overlap.query_end_position_in_read_;
    overlap.target_start_position_in_read_ = anchor.target_position_in_read_ < overlap.target_start_position_in_read_ ? anchor.target_position_in_read_ : overlap.target_start_position_in_read_;
    overlap.target_end_position_in_read_ = anchor.target_position_in_read_ > overlap.target_end_position_in_read_ ? anchor.target_position_in_read_ : overlap.target_end_position_in_read_;

            // If the target start position is greater than the target end position
        // We can safely assume that the query and target are template and
        // complement reads. TODO: Incorporate sketchelement direction value when
        // this is implemented
        if (overlap.target_start_position_in_read_ >
            overlap.target_end_position_in_read_)
        {
            overlap.relative_strand = RelativeStrand::Reverse;
            auto tmp                    = overlap.target_end_position_in_read_;
            overlap.target_end_position_in_read_ =
                overlap.target_start_position_in_read_;
            overlap.target_start_position_in_read_ = tmp;
        }
        else
        {
            overlap.relative_strand = RelativeStrand::Forward;
        }
}

__global__ void anchmers_to_overlaps(const Anchmer* anchmers, const int32_t* overlap_ends, const size_t n_anchmers, const Anchor* anchors, const size_t n_anchors, Overlap* overlaps, const size_t n_overlaps)
{
    // thread ID, which is used to index into the anchmers array
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (d_tid < n_anchmers){
        // printf("%u %d\n", anchmers[d_tid].n_anchors, anchmers[d_tid].n_chains);
        for (std::size_t i = 0; i < anchmers[d_tid].n_chains; ++i){
            std::size_t overlap_index = overlap_ends[d_tid] - anchmers[d_tid].n_chains + i;
            overlaps[overlap_index].query_start_position_in_read_ = 2147483647;
            overlaps[overlap_index].query_end_position_in_read_ = 0;
            overlaps[overlap_index].target_start_position_in_read_ = 2147483647;
            overlaps[overlap_index].target_end_position_in_read_ = 0;
            overlaps[overlap_index].num_residues_ = 0;
            for (std::size_t j = 0; j < anchmers[d_tid].n_anchors; ++j){
                add_anchor_to_overlap(anchors[d_tid * 10 + j], overlaps[overlap_index]);
                ++overlaps[overlap_index].num_residues_;
            }

        }
    }

}

__global__  void
generate_anchmers(const Anchor* d_anchors, const size_t n_anchors, Anchmer* anchmers, const uint8_t anchmer_size)
{

    // thread ID, which is used to index into the Anchors array
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // First index within the anchors array for this Anchmer
    std::size_t first_anchor_index = d_tid * anchmer_size;

    // Initialize Anchmer fields
    anchmers[d_tid].n_anchors = 0;
    anchmers[d_tid].n_chains = 0;
    std::int32_t current_chain = 1;
    for (int i = 0; i < 10; ++i){
        anchmers[d_tid].chain_id[i] = 0;
    }
    anchmers[d_tid].chain_id[0] = current_chain;
    anchmers[d_tid].n_chains = 1;
    // end intialization

    /**
    * Iterate through the anchors within this thread's range (first_anchor_index -> first_anchor_index + anchmer_size (or the end of the Anchors array))
    * For each anchor
    *   if the anchor has not been chained to another anchor, create a new chain (by incrementing the chain ID) and increment the number of chains in the Anchmer
    *   
    */
    for (std::size_t i = 0; i < anchmer_size; ++i){
        std::size_t global_anchor_index = first_anchor_index + i;
        if (global_anchor_index < n_anchors){
            ++(anchmers[d_tid].n_anchors);
            anchmers[d_tid].n_chains = anchmers[d_tid].chain_id[i] == 0 ? anchmers[d_tid].n_chains + 1 : anchmers[d_tid].n_chains;
            //Label the anchor with its chain ID
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

    //std::vector<Anchmer> anchmers(n_anchmers);
    device_buffer<Anchmer> d_anchmers(n_anchmers, _allocator, _cuda_stream);

    // Stage one: generate anchmers
    generate_anchmers<<<(n_anchmers / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchors.data(), n_anchors, d_anchmers.data(), anchors_per_anchmer);

    //cudautils::device_copy_n(d_anchmers.data(), d_anchmers.size(), anchmers.data(), _cuda_stream);

    // for (auto a : anchmers){
    //     std::cout << a.n_anchors << " " << static_cast<int16_t>(a.n_chains) << std::endl;
    //     for (std::size_t i = 0; i < a.n_anchors; ++i){
    //         std::cout << static_cast<int16_t>(a.chain_id[i]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Stage 2: Given a buffer of anchmers, generate overlaps within each anchmer.
    // Anchmers may contain between 1 and anchors_per_anchmer overlaps

    // Calculate the number of overlaps needed for the initial generation.
    // This is equal to the sum of each anchmer's n_chains value.
    //std::int32_t n_initial_overlaps = accumulate(begin(anchmers), end(anchmers), 0, [](std::int32_t sum, const Anchmer& anchmer){return sum + anchmer.n_chains;});

    // Transform each anchmer's n_chains value into a device vector so we can calculate a prefix
    // sum (which will give us the mapping between anchmer -> index in overlaps array)

    AnchmerCountChainsOp anchmer_chain_count_op;
    cub::TransformInputIterator<int32_t, AnchmerCountChainsOp, Anchmer*>d_chain_counts(d_anchmers.data(), anchmer_chain_count_op);
    
    device_buffer<int32_t> d_overlap_ends(n_anchmers, _allocator, _cuda_stream);

    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage     = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
         d_chain_counts, d_overlap_ends.data(), n_anchmers, _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
        d_chain_counts, d_overlap_ends.data(), n_anchmers, _cuda_stream);

    // Holds the last prefix sum in the overlap_ends vector.
    // This value is the total number of overlaps
    int32_t n_initial_overlaps = cudautils::get_value_from_device(d_overlap_ends.data() + n_anchmers - 1, _cuda_stream);

    device_buffer<Overlap> d_initial_overlaps (n_initial_overlaps, _allocator, _cuda_stream);
    std::cerr << "Generating " << n_initial_overlaps << " overlaps from " << n_anchmers << " anchmers..." << std::endl;
    fused_overlaps.resize(n_initial_overlaps);
    
    // Generate overlaps within each anchmer, filling the overlaps buffer
    anchmers_to_overlaps<<<(n_anchmers / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchmers.data(), d_overlap_ends.data(), n_anchmers,
     d_anchors.data(), n_anchors,
     d_initial_overlaps.data(), n_initial_overlaps);
    cudautils::device_copy_n(d_initial_overlaps.data(), d_initial_overlaps.size(), fused_overlaps.data(), _cuda_stream);


    // Sort overlaps

    // Perform a round of overlap fusion

    // Perform a round of overlap filtering

        // This is not completely necessary, but if removed one has to make sure that the next step
    // uses the same stream or that sync is done in caller
    GW_CU_CHECK_ERR(cudaStreamSynchronize(_cuda_stream));
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
