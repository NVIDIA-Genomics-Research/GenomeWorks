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

//
//          36a32532-4135-4ffe-a346-0b1b08c1b747   6370    173     683     -       fedde900-1485-42a2-8adb-b7a30dcf82fe    10014   91      620     105     529     0       minimap2        not_in_cm
//         da4230aa-e79e-4a5f-9738-c272aad98a82   8554    74      293     +       ebb4ceeb-76f1-454a-84d9-a65cfcb3a9fa    1824    1572    1801    114     231     0       minimap2        not_in_cm

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

#define MAX_ANCHMER_WINDOW 10
#define MAX_OVERLAPMER_WINDOW 20

struct Anchmer
{
    std::int8_t n_chained_anchors[MAX_ANCHMER_WINDOW] = {0};
    std::int8_t chain_id[MAX_ANCHMER_WINDOW]          = {0};
    std::int8_t n_chains                              = 0;
    std::int32_t n_anchors                            = 0;
};

struct Overlapmer
{
    std::int8_t n_chained_anchors[MAX_OVERLAPMER_WINDOW] = {0};
    std::int8_t chain_id[MAX_OVERLAPMER_WINDOW]          = {0};
    std::int8_t n_chains                                 = 0;
    std::int8_t n_overlaps                               = 0;
};

struct ChainPiece
{
    Overlap overlap;
    __device__ ChainPiece() {}
};

struct OverlapToChainPieceOp
{

    __device__ __forceinline__ ChainPiece operator()(const Overlap& a) const
    {
        ChainPiece c;
        c.overlap = a;
        return c;
    }
};

struct QueryTargetPair
{
    Overlap overlap;
    __device__ QueryTargetPair() {}
};

struct OverlapToQueryTargetPairOp
{
    __device__ __forceinline__ QueryTargetPair operator()(const Overlap& a) const
    {
        QueryTargetPair p;
        p.overlap = a;
        return p;
    }
};

__device__ bool operator==(const QueryTargetPair& a, const QueryTargetPair& b)
{
    return a.overlap.query_read_id_ == b.overlap.query_read_id_ && a.overlap.target_read_id_ == b.overlap.target_read_id_;
}

__device__ bool
operator==(const ChainPiece& a, const ChainPiece& b)
{
    const bool q_adjacent = abs(int(b.overlap.query_start_position_in_read_) - int(a.overlap.query_end_position_in_read_)) < 5000;
    const bool t_adjacent = a.overlap.relative_strand == RelativeStrand::Forward ? abs(int(a.overlap.target_end_position_in_read_) - int(b.overlap.target_start_position_in_read_)) < 5000 : abs(int(b.overlap.target_start_position_in_read_) - int(a.overlap.target_end_position_in_read_)) < 5000;
    return a.overlap.query_read_id_ == b.overlap.query_read_id_ &&
           a.overlap.target_read_id_ == b.overlap.target_read_id_ &&
           a.overlap.relative_strand == b.overlap.relative_strand &&
           t_adjacent && q_adjacent;
}
__device__ bool operator==(const Overlap& a,
                           const Overlap& b)
{
    bool same_strand   = a.relative_strand == b.relative_strand;
    bool identical_ids = a.query_read_id_ == b.query_read_id_ && a.target_read_id_ == b.target_read_id_;
    // bool q_ends_overlap;
    // bool t_end_overlap;
    position_in_read_t q_gap = abs((int)b.query_start_position_in_read_ - (int)a.query_end_position_in_read_);
    position_in_read_t t_gap = abs((int)b.target_start_position_in_read_ - (int)a.target_end_position_in_read_);
    bool gap_match           = q_gap < 150 && t_gap < 150;
    bool gap_ratio_okay      = float(min(q_gap, t_gap) / max(q_gap, t_gap)) < 0.8;

    //return identical_ids && same_strand && (gap_match || gap_ratio_okay);

    return identical_ids && same_strand && (gap_match);
}

__device__ bool operator==(const Anchor& lhs,
                           const Anchor& rhs)
{
    auto score_threshold = 1;

    // Very simple scoring function to quantify quality of overlaps.
    auto score = 1;

    if (abs(int(rhs.query_position_in_read_) - int(lhs.query_position_in_read_)) <= 50 and
        abs(int(rhs.target_position_in_read_) - int(lhs.target_position_in_read_)) <= 50)
        score = 2;
    return ((lhs.query_read_id_ == rhs.query_read_id_) &&
            (lhs.target_read_id_ == rhs.target_read_id_) &&
            score > score_threshold);
}

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
        c.relative_strand                = b.relative_strand;
        c.query_start_position_in_read_  = min(a.query_start_position_in_read_, b.query_start_position_in_read_);
        c.query_end_position_in_read_    = max(a.query_end_position_in_read_, b.query_end_position_in_read_);
        c.target_start_position_in_read_ = min(a.target_start_position_in_read_, b.target_start_position_in_read_);
        c.target_end_position_in_read_   = max(a.target_end_position_in_read_, b.target_end_position_in_read_);
        c.num_residues_                  = a.num_residues_ + b.num_residues_;
        return c;
    }
};

struct MergeChainPiecesOp
{
    __device__ __forceinline__ ChainPiece operator()(const ChainPiece& a, const ChainPiece& b)
    {
        Overlap c;
        c.query_read_id_                 = a.overlap.query_read_id_;
        c.target_read_id_                = a.overlap.target_read_id_;
        c.relative_strand                = a.overlap.relative_strand;
        c.query_start_position_in_read_  = min(a.overlap.query_start_position_in_read_, b.overlap.query_start_position_in_read_);
        c.query_end_position_in_read_    = max(a.overlap.query_end_position_in_read_, b.overlap.query_end_position_in_read_);
        c.target_start_position_in_read_ = min(a.overlap.target_start_position_in_read_, b.overlap.target_start_position_in_read_);
        c.target_end_position_in_read_   = max(a.overlap.target_end_position_in_read_, b.overlap.target_end_position_in_read_);
        c.num_residues_                  = a.overlap.num_residues_ + b.overlap.num_residues_;
        ChainPiece c_p;
        c_p.overlap = c;
        return c_p;
    }
};

struct DecrementerOp
{
    __host__ __device__ __forceinline__ std::size_t operator()(const std::size_t& val)
    {
        return val - 1;
    }
};

__global__ void mask_overlaps(Overlap* overlaps, std::size_t n_overlaps, bool* mask,
                              const std::size_t min_overlap_length,
                              const std::size_t min_residues,
                              const std::size_t min_bases_per_residue,
                              const bool all_to_all,
                              const bool filter_self_mappings)
{
    std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        position_in_read_t overlap_query_length  = overlaps[d_tid].query_end_position_in_read_ - overlaps[d_tid].query_start_position_in_read_;
        position_in_read_t overlap_target_length = overlaps[d_tid].target_end_position_in_read_ - overlaps[d_tid].target_start_position_in_read_;
        const bool mask_self_self                = overlaps[d_tid].query_read_id_ == overlaps[d_tid].target_read_id_ && all_to_all && filter_self_mappings;
        auto query_bases_per_residue             = overlap_query_length / overlaps[d_tid].num_residues_;
        auto target_bases_per_residue            = overlap_target_length / overlaps[d_tid].num_residues_;
        mask[d_tid]                              = overlap_query_length >= min_overlap_length & overlap_target_length >= min_overlap_length;
        mask[d_tid] &= overlaps[d_tid].num_residues_ >= min_residues;
        mask[d_tid] &= !mask_self_self;
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

__global__ void convert_offsets_to_ends(std::int32_t* starts, std::int32_t* lengths, std::int32_t* ends, std::int32_t n_starts)
{
    std::int32_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
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
                                   std::int32_t* starts, std::int32_t* ends, std::size_t n_runs,
                                   Overlap* fused_overlaps)
{
    std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_runs)
    {
        fused_overlaps[d_tid] = overlaps[starts[d_tid]];
        if (starts[d_tid] != ends[d_tid])
        {
            for (std::size_t i = starts[d_tid] + 1; i < ends[d_tid]; ++i)
            {
                fused_overlaps[d_tid] = merge_helper(fused_overlaps[d_tid], overlaps[i]);
            }
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

__device__ __forceinline__ bool overlaps_mergable(const Overlap& a, const Overlap& b, std::int32_t max_dist)
{
    const bool q_adjacent = abs(int(b.query_start_position_in_read_) - int(a.query_end_position_in_read_)) <= max_dist;
    const bool t_adjacent = abs(int(a.target_end_position_in_read_) - int(b.target_start_position_in_read_)) <= max_dist || abs(int(a.target_start_position_in_read_) - int(b.target_end_position_in_read_)) <= max_dist;
    return a.query_read_id_ == b.query_read_id_ &&
           a.target_read_id_ == b.target_read_id_ &&
           a.relative_strand == b.relative_strand &&
           t_adjacent && q_adjacent;
}

__global__ void chain_overlaps_in_window(Overlap* overlaps,
                                         const std::int32_t n_overlaps,
                                         const std::int32_t n_overlapmers,
                                         bool* overlap_mask,
                                         const std::int32_t overlapmer_size, const std::int32_t max_dist)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlapmers)
    {
        std::int32_t first_overlap_index = d_tid * overlapmer_size;
        for (std::size_t i = 0; i < overlapmer_size && i < n_overlaps; ++i)
        {
            std::size_t global_overlap_index   = first_overlap_index + i;
            overlap_mask[global_overlap_index] = true;
            if (global_overlap_index < n_overlaps)
            {

                std::size_t j = i + 1;
                while (j < overlapmer_size && j + first_overlap_index < n_overlaps)
                {
                    // if (overlaps[global_overlap_index] == overlaps[first_overlap_index + j])
                    if (overlaps_mergable(overlaps[global_overlap_index], overlaps[first_overlap_index + j], max_dist))
                    {
                        overlaps[first_overlap_index + j]     = merge_helper(overlaps[first_overlap_index + j], overlaps[first_overlap_index + i]);
                        overlap_mask[first_overlap_index + i] = false;
                    }
                    ++j;
                }
            }
        }
    }
}

// void chain_overlaps(Overlap* overlaps,
//                     bool* overlap_mask,
//                     Overlap* chains,
//                     std::size_t* d_num_filtered_overlaps,
//                     std::int32_t n_overlaps,
//                     const std::int32_t overlapmer_size,
//                     std::int32_t max_dist)
// {
//     std::int32_t num_overlapmers = (n_overlaps / overlapmer_size) + 1;
//     chain_overlaps_in_window<<<(n_chains / overlapmer_window_size) + 1, block_size, 0, _cuda_stream>>>(overlaps.data(),
//                                                                                                        n_overlaps,
//                                                                                                        (n_overlaps / overlapmer_size) + 1,
//                                                                                                        overlap_mask.data(),
//                                                                                                        overlapmer_size, max_dist);

//     void* d_temp_storage      = nullptr;
//     size_t temp_storage_bytes = 0;
//     cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, overlaps.data(),
//                                d_initial_overlap_mask.data(),
//                                d_filtered_overlaps.data(),
//                                d_num_filtered_overlaps,
//                                n_chains,
//                                _cuda_stream);
//     d_temp_buf.clear_and_resize(temp_storage_bytes);
//     d_temp_storage = d_temp_buf.data();
//     cub::DeviceSelect::Flagged(d_temp_storage,
//                                temp_storage_bytes,
//                                d_fused_chains.data(),
//                                d_initial_overlap_mask.data(),
//                                d_filtered_overlaps.data(),
//                                d_num_filtered_overlaps,
//                                n_chains,
//                                _cuda_stream);
// }

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
    const std::int32_t anchors_per_anchmer    = MAX_ANCHMER_WINDOW;
    const std::int32_t overlapmer_window_size = MAX_OVERLAPMER_WINDOW;
    std::size_t n_anchors                     = d_anchors.size();
    std::size_t n_anchmers                    = (d_anchors.size() / anchors_per_anchmer) + 1;
    std::int32_t block_size                   = 32;

    //std::vector<Anchmer> anchmers(n_anchmers);
    device_buffer<Anchmer> d_anchmers(n_anchmers, _allocator, _cuda_stream);

    // Stage one: generate anchmers
    generate_anchmers<<<(n_anchmers / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchors.data(), n_anchors, d_anchmers.data(), anchors_per_anchmer);

#ifdef DEBUG
    cudautils::device_copy_n(d_anchmers.data(), d_anchmers.size(), anchmers.data(), _cuda_stream);

    for (auto a : anchmers)
    {
        std::cout << a.n_anchors << " " << static_cast<int16_t>(a.n_chains) << std::endl;
        for (std::size_t i = 0; i < a.n_anchors; ++i)
        {
            std::cout << static_cast<int16_t>(a.chain_id[i]) << " ";
        }
        std::cout << std::endl;
    }
#endif

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
    cub::DeviceScan::InclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  d_chain_counts,
                                  d_overlap_ends.data(),
                                  n_anchmers,
                                  _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceScan::InclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  d_chain_counts,
                                  d_overlap_ends.data(),
                                  n_anchmers,
                                  _cuda_stream);

    // Holds the last prefix sum in the overlap_ends vector.
    // This value is the total number of overlaps
    int32_t n_initial_overlaps = cudautils::get_value_from_device(d_overlap_ends.data() + n_anchmers - 1, _cuda_stream);

    std::cerr << "Generating " << n_initial_overlaps << " initial overlaps from " << n_anchmers << " anchmers..." << std::endl;

    // Initialize overlaps to hold default values
    device_buffer<Overlap> d_initial_overlaps(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<Overlap> d_filtered_overlaps(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<bool> d_initial_overlap_mask(n_initial_overlaps, _allocator, _cuda_stream);

    initialize_overlaps_array<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_initial_overlaps.data(), n_initial_overlaps);

    // Generate overlaps within each anchmer, filling the overlaps buffer
    anchmers_to_overlaps<<<(n_anchmers / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchmers.data(), d_overlap_ends.data(), n_anchmers,
                                                                                         d_anchors.data(), n_anchors, d_initial_overlaps.data(), n_initial_overlaps);

    /** Initial overlaps have now been generated **/
    device_buffer<Overlap> d_init_chains(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<int32_t> d_init_starts(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<int32_t> d_init_chain_lengths(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<int32_t> d_init_ends(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<std::size_t> d_init_num_chains(1, _allocator, _cuda_stream);
    // Perform a round of RLE on the overlaps
    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_initial_overlaps.data(),
                                       d_init_chains.data(),
                                       d_init_chain_lengths.data(),
                                       d_init_num_chains.data(),
                                       n_initial_overlaps);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_initial_overlaps.data(),
                                       d_init_chains.data(),
                                       d_init_chain_lengths.data(),
                                       d_init_num_chains.data(),
                                       n_initial_overlaps);
    std::size_t n_chains = cudautils::get_value_from_device(d_init_num_chains.data(), _cuda_stream);
    d_temp_storage       = nullptr;
    temp_storage_bytes   = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_init_chain_lengths.data(), d_init_starts.data(),
                                  n_chains, _cuda_stream);
    // allocate temporary storage
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_init_chain_lengths.data(), d_init_starts.data(),
                                  n_chains, _cuda_stream);

    convert_offsets_to_ends<<<(n_chains / block_size) + 1, block_size, 0, _cuda_stream>>>(d_init_starts.data(),
                                                                                          d_init_chain_lengths.data(),
                                                                                          d_init_ends.data(), n_chains);
    d_initial_overlap_mask.clear_and_resize(n_chains);
    device_buffer<Overlap> d_fused_chains(n_chains, _allocator, _cuda_stream);
    merge_overlap_runs<<<(n_chains / block_size) + 1, block_size, 0, _cuda_stream>>>(d_initial_overlaps.data(), d_init_starts.data(), d_init_ends.data(), n_chains, d_fused_chains.data());

    std::cerr << "Merged " << n_chains << " initial chains." << std::endl;
    d_initial_overlap_mask.clear_and_resize(n_chains);
    /*** First round of local overlap chaining ***/
    chain_overlaps_in_window<<<(n_chains / overlapmer_window_size) + 1, block_size, 0, _cuda_stream>>>(d_fused_chains.data(),
                                                                                                       n_chains,
                                                                                                       (n_chains / overlapmer_window_size) + 1,
                                                                                                       d_initial_overlap_mask.data(),
                                                                                                       overlapmer_window_size, 150);
    std::cerr << "First round of superchaining complete." << std::endl;

    // // Remove overlaps that were masked during chaining
    device_buffer<size_t> d_num_filtered_overlaps(1, _allocator, _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_chains.data(),
                               d_initial_overlap_mask.data(),
                               d_filtered_overlaps.data(),
                               d_num_filtered_overlaps.data(),
                               n_chains,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage,
                               temp_storage_bytes,
                               d_fused_chains.data(),
                               d_initial_overlap_mask.data(),
                               d_filtered_overlaps.data(),
                               d_num_filtered_overlaps.data(),
                               n_chains,
                               _cuda_stream);

    std::size_t n_filtered_overlaps = cudautils::get_value_from_device(d_num_filtered_overlaps.data(), _cuda_stream);
    std::cerr << n_filtered_overlaps << " remain after initial chaining round." << std::endl;

    d_initial_overlap_mask.clear_and_resize(n_filtered_overlaps);
    /*** Second round of local overlap chaining ***/
    chain_overlaps_in_window<<<(n_chains / overlapmer_window_size) + 1, block_size, 0, _cuda_stream>>>(d_filtered_overlaps.data(),
                                                                                                       n_filtered_overlaps,
                                                                                                       (n_filtered_overlaps / overlapmer_window_size) + 1,
                                                                                                       d_initial_overlap_mask.data(),
                                                                                                       overlapmer_window_size, 1000);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_filtered_overlaps.data(),
                               d_initial_overlap_mask.data(),
                               d_fused_chains.data(),
                               d_num_filtered_overlaps.data(),
                               n_filtered_overlaps,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_filtered_overlaps.data(),
                               d_initial_overlap_mask.data(),
                               d_fused_chains.data(),
                               d_num_filtered_overlaps.data(),
                               n_filtered_overlaps,
                               _cuda_stream);
    n_filtered_overlaps = cudautils::get_value_from_device(d_num_filtered_overlaps.data(), _cuda_stream);
    std::cerr << n_filtered_overlaps << " remain after second chaining round." << std::endl;

    d_initial_overlap_mask.clear_and_resize(n_filtered_overlaps);
    mask_overlaps<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_fused_chains.data(), n_filtered_overlaps, d_initial_overlap_mask.data(), 10, 2, 0, all_to_all, false);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_chains.data(),
                               d_initial_overlap_mask.data(),
                               d_filtered_overlaps.data(),
                               d_num_filtered_overlaps.data(),
                               n_filtered_overlaps,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_chains.data(),
                               d_initial_overlap_mask.data(),
                               d_filtered_overlaps.data(),
                               d_num_filtered_overlaps.data(),
                               n_filtered_overlaps,
                               _cuda_stream);
    n_filtered_overlaps = cudautils::get_value_from_device(d_num_filtered_overlaps.data(), _cuda_stream);
    std::cerr << n_filtered_overlaps << " remain after post-secondary filtering." << std::endl;

    /*** Tertiary chaining ***/
    d_initial_overlap_mask.clear_and_resize(n_filtered_overlaps);
    d_fused_chains.clear_and_resize(n_filtered_overlaps);
    chain_overlaps_in_window<<<(n_chains / overlapmer_window_size) + 1, block_size, 0, _cuda_stream>>>(d_filtered_overlaps.data(),
                                                                                                       n_filtered_overlaps,
                                                                                                       (n_filtered_overlaps / overlapmer_window_size) + 1,
                                                                                                       d_initial_overlap_mask.data(),
                                                                                                       overlapmer_window_size, 5000);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_filtered_overlaps.data(),
                               d_initial_overlap_mask.data(),
                               d_fused_chains.data(),
                               d_num_filtered_overlaps.data(),
                               n_filtered_overlaps,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_filtered_overlaps.data(),
                               d_initial_overlap_mask.data(),
                               d_fused_chains.data(),
                               d_num_filtered_overlaps.data(),
                               n_filtered_overlaps,
                               _cuda_stream);
    n_filtered_overlaps = cudautils::get_value_from_device(d_num_filtered_overlaps.data(), _cuda_stream);
    std::cerr << "After long-chaining, " << n_filtered_overlaps << " remain." << std::endl;

    device_buffer<Overlap> d_final_overlaps(n_filtered_overlaps, _allocator, _cuda_stream);
    d_initial_overlap_mask.clear_and_resize(n_filtered_overlaps);
    mask_overlaps<<<(n_filtered_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_fused_chains.data(), n_chains, d_initial_overlap_mask.data(), min_overlap_len, min_residues, min_bases_per_residue, all_to_all, true);
    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_chains.data(),
                               d_initial_overlap_mask.data(),
                               d_final_overlaps.data(),
                               d_num_filtered_overlaps.data(),
                               n_filtered_overlaps,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_chains.data(),
                               d_initial_overlap_mask.data(),
                               d_final_overlaps.data(),
                               d_num_filtered_overlaps.data(),
                               n_filtered_overlaps,
                               _cuda_stream);
    n_filtered_overlaps = cudautils::get_value_from_device(d_num_filtered_overlaps.data(), _cuda_stream);
    std::cerr << "After final filtering, " << n_filtered_overlaps << " remain." << std::endl;

    fused_overlaps.resize(n_filtered_overlaps);
    cudautils::device_copy_n(d_final_overlaps.data(), n_filtered_overlaps, fused_overlaps.data(), _cuda_stream);

    // d_temp_storage     = nullptr;
    // temp_storage_bytes = 0;
    // cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
    //                               d_chain_lengths.data(), d_chain_starts.data(),
    //                               n_chains, _cuda_stream);

    // // allocate temporary storage
    // d_temp_buf.clear_and_resize(temp_storage_bytes);
    // d_temp_storage = d_temp_buf.data();

    // cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
    //                               d_chain_lengths.data(), d_chain_starts.data(),
    //                               n_chains, _cuda_stream);

    // device_buffer<Overlap> d_fused_chains(n_chains, _allocator, _cuda_stream);
    // device_buffer<std::size_t> d_chain_ends(n_overlap_runs, _allocator, _cuda_stream);
    // convert_offsets_to_ends<<<(n_overlap_runs / block_size) + 1, block_size, 0, _cuda_stream>>>(d_chain_starts.data(), d_chain_lengths.data(), d_chain_ends.data(), n_overlap_runs);
    // merge_overlap_runs<<<(n_chains) + 1, block_size, 0, _cuda_stream>>>(d_fused_overlaps.data(), d_chain_starts.data(), d_chain_ends.data(), n_chains, d_fused_chains.data());

    // d_initial_overlap_mask.clear_and_resize(n_chains);
    // mask_overlaps<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_fused_chains.data(), n_chains, d_initial_overlap_mask.data(), min_overlap_len, min_residues, min_bases_per_residue, all_to_all, true);

    // device_buffer<Overlap> d_final_overlaps(n_chains, _allocator, _cuda_stream);
    // device_buffer<size_t> d_num_final_overlaps(1, _allocator, _cuda_stream);

    // d_temp_storage     = nullptr;
    // temp_storage_bytes = 0;
    // cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_chains.data(),
    //                            d_initial_overlap_mask.data(),
    //                            d_final_overlaps.data(),
    //                            d_num_final_overlaps.data(),
    //                            n_chains,
    //                            _cuda_stream);
    // d_temp_buf.clear_and_resize(temp_storage_bytes);
    // d_temp_storage = d_temp_buf.data();
    // cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_fused_chains.data(),
    //                            d_initial_overlap_mask.data(),
    //                            d_final_overlaps.data(),
    //                            d_num_final_overlaps.data(),
    //                            n_chains,
    //                            _cuda_stream);
    // std::size_t n_final_overlaps = cudautils::get_value_from_device(d_num_final_overlaps.data(), _cuda_stream);
    // std::cerr << "Produced " << n_final_overlaps << " final overlaps; " << n_overlap_runs - n_final_overlaps << " overlaps removed by filtering." << std::endl;

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
