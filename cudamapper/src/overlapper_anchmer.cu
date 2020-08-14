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

#define MAX_ANCHMER_WINDOW 10

struct Anchmer
{
    std::int8_t n_chained_anchors[MAX_ANCHMER_WINDOW] = {0};
    std::int8_t chain_id[MAX_ANCHMER_WINDOW]          = {0};
    std::int8_t n_chains                              = 0;
    std::int32_t n_anchors                            = 0;
};

__device__ bool operator==(const Overlap& a,
                           const Overlap& b)
{
    bool same_strand   = a.relative_strand == b.relative_strand;
    bool identical_ids = a.query_read_id_ == b.query_read_id_ && a.target_read_id_ == b.target_read_id_;
    // bool q_ends_overlap;
    // bool t_end_overlap;
    position_in_read_t q_gap = abs((int)b.query_start_position_in_read_ - (int)a.query_end_position_in_read_);
    position_in_read_t t_gap = abs((int)b.target_start_position_in_read_ - (int)a.target_end_position_in_read_);
    bool gap_match           = q_gap < 300 && t_gap < 300;
    bool gap_ratio_okay      = float(min(q_gap, t_gap) / max(q_gap, t_gap)) < 0.8;

    return identical_ids && same_strand && (gap_match || gap_ratio_okay);
}

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
        return static_cast<int32_t>(a.n_chains + b.n_chains);
    }
};

struct AnchmerCountChainsOp
{

    AnchmerCountChainsOp()
    {
    }

    __host__ __device__ __forceinline__
        std::int32_t
        operator()(const Anchmer& a) const
    {
        return static_cast<int32_t>(a.n_chains);
    }
};

struct MergeOverlapRunOp
{
    __device__ __forceinline__ Overlap operator()(const Overlap& a, const Overlap& b)
    {
        Overlap c;
        c.query_read_id_                 = a.query_read_id_;
        c.target_read_id_                = a.target_read_id_;
        c.relative_strand                = a.relative_strand;
        c.query_start_position_in_read_  = min(a.query_start_position_in_read_, b.query_start_position_in_read_);
        c.query_end_position_in_read_    = max(a.query_end_position_in_read_, b.query_end_position_in_read_);
        c.target_start_position_in_read_ = min(a.target_start_position_in_read_, b.target_start_position_in_read_);
        c.target_end_position_in_read_   = max(a.target_end_position_in_read_, b.target_end_position_in_read_);
        c.num_residues_                  = a.num_residues_ + b.num_residues_;
        return c;
    }
};

struct DecrementerOp
{
    __host__ __device__ __forceinline__ std::size_t operator() (const std::size_t& val)
    {
        return val - 1;
    }
};

__global__ void mask_overlaps(Overlap* overlaps, std::size_t n_overlaps, bool* mask, const std::size_t min_overlap_length, const std::size_t min_residues, const std::size_t min_bases_per_residue)
{
    std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        position_in_read_t overlap_query_length  = overlaps[d_tid].query_end_position_in_read_ - overlaps[d_tid].query_start_position_in_read_;
        position_in_read_t overlap_target_length = overlaps[d_tid].target_end_position_in_read_ - overlaps[d_tid].target_start_position_in_read_;
        auto query_bases_per_residue = overlap_query_length / overlaps[d_tid].num_residues_;
        auto target_bases_per_residue = overlap_target_length / overlaps[d_tid].num_residues_;
        mask[d_tid]                              = overlap_query_length > min_overlap_length & overlap_target_length > min_overlap_length;
        mask[d_tid] &= overlaps[d_tid].num_residues_ >= min_residues;
        //mask[d_tid] &= (query_bases_per_residue < min_bases_per_residue || target_bases_per_residue < min_bases_per_residue);
    }
}

__global__ void finalize_overlaps(Overlap* overlaps, const std::size_t n_overlaps)
{
    std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        if (overlaps[d_tid].target_start_position_in_read_ > overlaps[d_tid].target_end_position_in_read_)
        {
            overlaps[d_tid].relative_strand                = RelativeStrand::Reverse;
            auto tmp                                       = overlaps[d_tid].target_start_position_in_read_;
            overlaps[d_tid].target_start_position_in_read_ = overlaps[d_tid].target_end_position_in_read_;
            overlaps[d_tid].target_end_position_in_read_   = tmp;
        }
    }
};

__global__ void convert_offsets_to_ends(std::size_t* starts, std::size_t* lengths, std::size_t* ends, std::size_t n_starts)
{
    std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_starts)
    {
        ends[d_tid] = starts[d_tid] + lengths[d_tid] - 1;
    }
}

__device__ __forceinline__ Overlap merge_helper(Overlap& a, Overlap& b)
{
        Overlap c;
        c.query_read_id_                 = a.query_read_id_;
        c.target_read_id_                = a.target_read_id_;
        c.relative_strand                = a.relative_strand;
        c.query_start_position_in_read_  = min(a.query_start_position_in_read_, b.query_start_position_in_read_);
        c.query_end_position_in_read_    = max(a.query_end_position_in_read_, b.query_end_position_in_read_);
        c.target_start_position_in_read_ = min(a.target_start_position_in_read_, b.target_start_position_in_read_);
        c.target_end_position_in_read_   = max(a.target_end_position_in_read_, b.target_end_position_in_read_);
        c.num_residues_                  = a.num_residues_ + b.num_residues_;
        return c;
}

__global__ void merge_overlap_runs(Overlap* overlaps,
     std::size_t* starts, std::size_t* ends, std::size_t n_runs,
    Overlap* fused_overlaps)
{
    std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_runs)
    {
        fused_overlaps[d_tid] = overlaps[starts[d_tid]];
        for (std::size_t i = starts[d_tid] + 1; i < ends[d_tid]; ++i){
            fused_overlaps[d_tid] = merge_helper(fused_overlaps[d_tid], overlaps[i]);
        }
    }
}

__device__ __forceinline__ void add_anchor_to_overlap(const Anchor& anchor, Overlap& overlap)
{
    overlap.query_read_id_                = anchor.query_read_id_;
    overlap.target_read_id_               = anchor.target_read_id_;
    overlap.query_start_position_in_read_ = min(anchor.query_position_in_read_, overlap.query_start_position_in_read_);
    overlap.query_end_position_in_read_   = max(anchor.query_position_in_read_, overlap.query_end_position_in_read_);

    // Handles whether the match is on the forward or reverse strand.
    // Requires anchors to be sorted in order
    // First by query_read_id, then target_read_id, then query_pos, then target_pos.
    // If these are sorted, adding an anchor to an overlap with at least one
    // anchor in it already will indicate whether the overlap is increasing on its start or end.
    // If the anchor falls before the target_start, we are on the reverse strand and we should use it to extend
    // the target_end_.
    // If the anchor falls after the target_start, we should extend the target_start
    // Since the anchors will be monotonically increasing or monotonically decreasing, this
    // should consistently head in the correct direction.

    if (overlap.num_residues_ == 0)
    {
        overlap.target_start_position_in_read_ = anchor.target_position_in_read_;
        overlap.target_end_position_in_read_   = anchor.target_position_in_read_;
    }
    else
    {
        bool on_forward_strand                 = anchor.target_position_in_read_ > overlap.target_start_position_in_read_;
        overlap.target_start_position_in_read_ = on_forward_strand ? overlap.target_start_position_in_read_ : anchor.target_position_in_read_;
        overlap.target_end_position_in_read_   = on_forward_strand ? anchor.target_position_in_read_ : overlap.target_end_position_in_read_;
        overlap.relative_strand                = on_forward_strand ? RelativeStrand::Forward : RelativeStrand::Reverse;
    }

    ++overlap.num_residues_;
}

__device__ __forceinline__ void init_overlap(Overlap& overlap)
{
    overlap.query_start_position_in_read_  = 4294967295;
    overlap.query_end_position_in_read_    = 0;
    overlap.target_start_position_in_read_ = 4294967295;
    overlap.target_end_position_in_read_   = 0;
    overlap.relative_strand                = RelativeStrand::Forward;
    overlap.num_residues_                  = 0;
}

__global__ void initialize_overlaps_array(Overlap* overlaps, const size_t n_overlaps)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        init_overlap(overlaps[d_tid]);
    }
}

__global__ void anchmers_to_overlaps(const Anchmer* anchmers, const int32_t* overlap_ends, const size_t n_anchmers, const Anchor* anchors, const size_t n_anchors, Overlap* overlaps, const size_t n_overlaps)
{
    // thread ID, which is used to index into the anchmers array
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (d_tid < n_anchmers)
    {
        for (std::size_t i = 0; i < anchmers[d_tid].n_anchors; ++i)
        {
            std::size_t overlap_index = overlap_ends[d_tid] - anchmers[d_tid].n_chains + anchmers[d_tid].chain_id[i];
            add_anchor_to_overlap(anchors[d_tid * MAX_ANCHMER_WINDOW + i], overlaps[overlap_index]);
        }
    }
}

__global__ void
generate_anchmers(const Anchor* d_anchors, const size_t n_anchors, Anchmer* anchmers, const uint8_t anchmer_size)
{

    // thread ID, which is used to index into the Anchors array
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // First index within the anchors array for this Anchmer
    std::size_t first_anchor_index = d_tid * anchmer_size;

    // Initialize Anchmer fields
    anchmers[d_tid].n_anchors  = 0;
    anchmers[d_tid].n_chains   = 0;
    std::int32_t current_chain = 1;
    for (int i = 0; i < MAX_ANCHMER_WINDOW; ++i)
    {
        anchmers[d_tid].chain_id[i] = 0;
    }
    anchmers[d_tid].chain_id[0] = current_chain;
    anchmers[d_tid].n_chains    = 1;
    // end intialization

    /**
    * Iterate through the anchors within this thread's range (first_anchor_index -> first_anchor_index + anchmer_size (or the end of the Anchors array))
    * For each anchor
    *   if the anchor has not been chained to another anchor, create a new chain (by incrementing the chain ID) and increment the number of chains in the Anchmer
    *   
    */
    for (std::size_t i = 0; i < anchmer_size; ++i)
    {
        std::size_t global_anchor_index = first_anchor_index + i;
        if (global_anchor_index < n_anchors)
        {
            ++(anchmers[d_tid].n_anchors);
            anchmers[d_tid].n_chains = anchmers[d_tid].chain_id[i] == 0 ? anchmers[d_tid].n_chains + 1 : anchmers[d_tid].n_chains;
            //Label the anchor with its chain ID
            anchmers[d_tid].chain_id[i] = anchmers[d_tid].chain_id[i] == 0 ? ++current_chain : anchmers[d_tid].chain_id[i];

            std::size_t j = i + 1;
            while (j < anchmer_size && j + first_anchor_index < n_anchors)
            {
                if (d_anchors[global_anchor_index] == d_anchors[first_anchor_index + j])
                {
                    anchmers[d_tid].chain_id[j] = anchmers[d_tid].chain_id[i];
                }
                ++j;
            }
        }
    }
}

__global__ void decrementer_kernel(std::size_t* vals, const std::size_t n_vals)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_vals)
    {
        vals[d_tid] = vals[d_tid] - 1;
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
    const std::int32_t anchors_per_anchmer = MAX_ANCHMER_WINDOW;
    std::size_t n_anchors                  = d_anchors.size();
    std::size_t n_anchmers                 = (d_anchors.size() / anchors_per_anchmer) + 1;
    std::int32_t block_size                = 32;

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

    // Transform each anchmer's n_chains value into a device vector so we can calculate a prefix
    // sum (which will give us the mapping between anchmer -> index in overlaps array)

    AnchmerCountChainsOp anchmer_chain_count_op;
    cub::TransformInputIterator<int32_t, AnchmerCountChainsOp, Anchmer*> d_chain_counts(d_anchmers.data(), anchmer_chain_count_op);

    device_buffer<int32_t> d_overlap_ends(n_anchmers, _allocator, _cuda_stream);

    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
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

    device_buffer<Overlap> d_initial_overlaps(n_initial_overlaps, _allocator, _cuda_stream);
    std::cerr << "Generating " << n_initial_overlaps << " overlaps from " << n_anchmers << " anchmers..." << std::endl;

    // Initialize overlaps to hold default values
    initialize_overlaps_array<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_initial_overlaps.data(), n_initial_overlaps);

    // Generate overlaps within each anchmer, filling the overlaps buffer
    anchmers_to_overlaps<<<(n_anchmers / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchmers.data(), d_overlap_ends.data(), n_anchmers,
                                                                                         d_anchors.data(), n_anchors, d_initial_overlaps.data(), n_initial_overlaps);
    //finalize_overlaps<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_initial_overlaps.data(), n_initial_overlaps);

    // First overlap filtering stage
    // Remove short overlaps (length < 5bp)
    device_buffer<bool> d_initial_overlap_mask(n_initial_overlaps, _allocator, _cuda_stream);
    mask_overlaps<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_initial_overlaps.data(), n_initial_overlaps, d_initial_overlap_mask.data(), 5, 0, 0);

    device_buffer<Overlap> d_filtered_overlaps(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<size_t> d_num_filtered_overlaps(1, _allocator, _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_initial_overlaps.data(),
                               d_initial_overlap_mask.data(),
                               d_filtered_overlaps.data(),
                               d_num_filtered_overlaps.data(),
                               n_initial_overlaps,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage,
                               temp_storage_bytes,
                               d_initial_overlaps.data(),
                               d_initial_overlap_mask.data(),
                               d_filtered_overlaps.data(),
                               d_num_filtered_overlaps.data(),
                               n_initial_overlaps,
                               _cuda_stream);
    std::size_t n_filtered_overlaps = cudautils::get_value_from_device(d_num_filtered_overlaps.data(), _cuda_stream);

    device_buffer<std::size_t> d_num_overlap_runs(1, _allocator, _cuda_stream);
    device_buffer<std::size_t> d_overlap_run_offsets(n_filtered_overlaps, _allocator, _cuda_stream);
    device_buffer<std::size_t> d_overlap_run_lengths(n_filtered_overlaps, _allocator, _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage,
                                               temp_storage_bytes,
                                               d_filtered_overlaps.data(),
                                               d_overlap_run_offsets.data(),
                                               d_overlap_run_lengths.data(),
                                               d_num_overlap_runs.data(),
                                               n_filtered_overlaps, _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceRunLengthEncode::NonTrivialRuns(d_temp_storage,
                                               temp_storage_bytes,
                                               d_filtered_overlaps.data(),
                                               d_overlap_run_offsets.data(),
                                               d_overlap_run_lengths.data(),
                                               d_num_overlap_runs.data(),
                                               n_filtered_overlaps, _cuda_stream);
    std::size_t n_overlap_runs = cudautils::get_value_from_device(d_num_overlap_runs.data(), _cuda_stream);

    std::cerr << "Found " << n_overlap_runs << " runs of non-trivial overlaps." << std::endl;
    device_buffer<Overlap> d_fused_overlaps(n_overlap_runs, _allocator, _cuda_stream);

    // Transform the overlap_run_lengths vector into a vector of overlap ends.
    // We'll use a prefix scan but it may not be the most efficient method.
    // TODO: explore not using a prefix scan
    device_buffer<std::size_t> d_overlap_run_ends(n_overlap_runs, _allocator, _cuda_stream);
    convert_offsets_to_ends<<<(n_overlap_runs / block_size)+1, block_size, 0, _cuda_stream>>>(d_overlap_run_offsets.data(), d_overlap_run_lengths.data(), d_overlap_run_ends.data(), n_overlap_runs);
    // d_temp_storage           = nullptr;
    // temp_storage_bytes = 0;
    // cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
    //                               d_overlap_run_lengths.data(), d_overlap_run_ends.data(), n_overlap_runs, _cuda_stream);

    // d_temp_buf.clear_and_resize(temp_storage_bytes);
    // d_temp_storage = d_temp_buf.data();

    // cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
    //                               d_overlap_run_lengths.data(), d_overlap_run_ends.data(), n_overlap_runs, _cuda_stream);
    // DecrementerOp decop;
    // cub::TransformInputIterator<std::size_t, DecrementerOp, std::size_t*> d_run_ends (d_overlap_run_ends.data(), decop);

    //decrementer_kernel<<<(n_overlap_runs / block_size)+1, block_size, 0, _cuda_stream>>>(d_overlap_run_ends.data(), n_overlap_runs);
    Overlap x;
    // Merge overlaps within a run
    MergeOverlapRunOp merge_op;
    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    // TODO: fix this so it uses the ends, and not the lengths, of the runs.
    // TODO: Grab the number of runs out
    // TODO: Tweak chaining logic in NonTrivialRuns to see if it improves accuracy.
    // cub::DeviceSegmentedReduce::Reduce(
    //     d_temp_storage,
    //     temp_storage_bytes,
    //     d_filtered_overlaps.data(),
    //     d_fused_overlaps.data(),
    //     n_overlap_runs,
    //     d_overlap_run_offsets.data(),
    //     d_overlap_run_ends.data(),
    //     merge_op,
    //     d_filtered_overlaps.data()[0],
    //     _cuda_stream);

    // std::vector<size_t> run_starts(n_overlap_runs);
    // cudautils::device_copy_n(d_overlap_run_offsets.data(), n_overlap_runs, run_starts.data(), _cuda_stream);
    // std::vector<size_t> run_ends(n_overlap_runs);
    // cudautils::device_copy_n(d_overlap_run_ends.data(), n_overlap_runs, run_ends.data(), _cuda_stream);
    // for (size_t i = 0; i < n_overlap_runs; ++i)
    // {
    //     std::cerr << run_starts[i] << " " << run_ends[i] << std::endl;
    // }
    merge_overlap_runs<<<(n_overlap_runs / block_size) + 1, block_size, 0, _cuda_stream>>>(d_filtered_overlaps.data(), d_overlap_run_offsets.data(), d_overlap_run_ends.data(), n_overlap_runs, d_fused_overlaps.data());
    std::cerr << "Fused overlap runs." << std::endl;


    d_initial_overlap_mask.clear_and_resize(n_overlap_runs);
    mask_overlaps<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_fused_overlaps.data(), n_overlap_runs, d_initial_overlap_mask.data(), min_overlap_len, min_residues, min_bases_per_residue);

    device_buffer<Overlap> d_final_overlaps(n_overlap_runs, _allocator, _cuda_stream);
    device_buffer<size_t> d_num_final_overlaps(1, _allocator, _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_overlaps.data(),
                               d_initial_overlap_mask.data(),
                               d_final_overlaps.data(),
                               d_num_final_overlaps.data(),
                               n_overlap_runs,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_overlaps.data(),
                               d_initial_overlap_mask.data(),
                               d_final_overlaps.data(),
                               d_num_final_overlaps.data(),
                               n_overlap_runs,
                               _cuda_stream);
    std::size_t n_final_overlaps = cudautils::get_value_from_device(d_num_final_overlaps.data(), _cuda_stream);


    fused_overlaps.resize(n_final_overlaps);
    cudautils::device_copy_n(d_final_overlaps.data(), n_final_overlaps, fused_overlaps.data(), _cuda_stream);

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
