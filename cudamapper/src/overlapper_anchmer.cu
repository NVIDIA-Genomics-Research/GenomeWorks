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
#define MAX_OVERLAPMER_WINDOW 10
#define INT32_INFINITY 1000000

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

struct AnchorToPosStruct
{

    __host__ __device__ __forceinline__ std::int32_t operator()(const Anchor* anchor) const
    {
        return static_cast<int32_t>(anchor->query_position_in_read_);
    }

    __host__ __device__ __forceinline__ std::int32_t operator()(const Anchor& anchor) const
    {
        return static_cast<int32_t>(anchor.query_position_in_read_);
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
    if (lhs.query_position_in_read_ == rhs.query_position_in_read_)
        score = 0;
    return ((lhs.query_read_id_ == rhs.query_read_id_) &&
            (lhs.target_read_id_ == rhs.target_read_id_) &&
            score > score_threshold);
}

struct AnchmerCountChainsOp
{

    AnchmerCountChainsOp()
    {
    }

    __host__ __device__ __forceinline__ std::int32_t operator()(const Anchmer& a) const
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
        //auto query_bases_per_residue             = overlap_query_length / overlaps[d_tid].num_residues_;
        //auto target_bases_per_residue            = overlap_target_length / overlaps[d_tid].num_residues_;
        mask[d_tid] &= overlap_query_length >= min_overlap_length & overlap_target_length >= min_overlap_length;
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
    c.relative_strand                = a.num_residues_ > b.num_residues_ ? a.relative_strand : b.relative_strand;
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

__device__ __forceinline__ bool point_contained(const position_in_read_t point, const position_in_read_t interval_start, const position_in_read_t interval_end)
{
    return point >= interval_start && point <= interval_end;
}

__device__ __forceinline__ position_in_read_t abs_diff_position(const position_in_read_t first, const position_in_read_t second)
{
    return abs(int(second) - int(first));
}

__device__ __forceinline__ bool point_adjacent(position_in_read_t point, position_in_read_t interval_start, position_in_read_t interval_end, position_in_read_t max_dist)
{
    return abs_diff_position(point, interval_start) < max_dist || abs_diff_position(point, interval_end) < max_dist;
}

__device__ __forceinline__ bool gaps_match(const Overlap& a, const Overlap& b, const double threshold)
{
    position_in_read_t q_gap = abs((int)b.query_start_position_in_read_ - (int)a.query_end_position_in_read_);
    position_in_read_t t_gap = abs((int)b.target_start_position_in_read_ - (int)a.target_end_position_in_read_);

    return min(q_gap, t_gap) / max(q_gap, t_gap) < threshold;
}

__device__ __forceinline__ bool exhaustive_overlap_compare(const Overlap& a, const Overlap& b)
{

    const bool same_strand = a.relative_strand == b.relative_strand;
    const bool query_same  = a.query_read_id_ == b.query_read_id_;
    const bool target_same = a.target_read_id_ == b.target_read_id_;

    const bool query_contained  = point_contained(b.query_start_position_in_read_, a.query_start_position_in_read_, a.query_end_position_in_read_) || point_contained(b.query_end_position_in_read_, a.query_start_position_in_read_, a.query_end_position_in_read_);
    const bool target_contained = point_contained(b.target_start_position_in_read_, a.target_start_position_in_read_, a.target_end_position_in_read_) || point_contained(b.target_end_position_in_read_, a.target_start_position_in_read_, a.target_end_position_in_read_);
    const bool query_adjacent   = point_adjacent(b.query_start_position_in_read_, a.query_start_position_in_read_, a.query_end_position_in_read_, 300) || point_adjacent(b.query_end_position_in_read_, a.query_start_position_in_read_, a.query_end_position_in_read_, 300);
    const bool target_adjacent  = point_adjacent(b.target_start_position_in_read_, a.target_start_position_in_read_, a.target_end_position_in_read_, 300) || point_adjacent(b.target_start_position_in_read_, a.target_start_position_in_read_, a.target_end_position_in_read_, 300);

    const bool query_target_gaps_match = gaps_match(a, b, 0.8);

    const bool positions_mergable = query_target_gaps_match || ((query_contained || query_adjacent) && (target_contained || target_adjacent));

    return query_same && target_same && same_strand && positions_mergable;
}

__global__ void merge_overlaps_in_query_target_pairs(Overlap* overlaps, std::int32_t* starts, std::int32_t* ends, bool* mask, const std::size_t n_qt_runs)
{
    std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_qt_runs)
    {
        if (starts[d_tid] != ends[d_tid])
        {
            for (std::int32_t ind = starts[d_tid]; ind < ends[d_tid]; ++ind)
            {
                mask[ind] = true;
            }
            for (std::int32_t i = starts[d_tid]; i < ends[d_tid]; ++i)
            {
                std::int32_t j = i + 1;
                while (j < ends[d_tid])
                {
                    if (mask[j] && exhaustive_overlap_compare(overlaps[i], overlaps[j]))
                    {
                        overlaps[i] = merge_helper(overlaps[i], overlaps[j]);
                        mask[j]     = false;
                    }
                    ++j;
                }
            }
        }
    }
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

__global__ void mask_anchor_repeat_runs(const Anchor* anchors,
                                        bool* anchor_mask,
                                        int32_t* run_starts,
                                        int32_t* run_lengths,
                                        const int32_t n_anchors,
                                        const int32_t n_runs,
                                        const int32_t min_repeat_length)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_runs)
    {
        if (run_lengths[d_tid] > min_repeat_length)
        {
            for (int32_t i = run_starts[d_tid]; i < run_starts[d_tid] + run_lengths[d_tid]; ++i)
            {
                anchor_mask[i] = false;
            }
        }
    }
}

void mask_repetitive_anchors_by_RLE(const device_buffer<Anchor>& d_anchors,
                                    device_buffer<bool>& d_mask,
                                    const size_t n_anchors,
                                    const int32_t max_placements,
                                    DefaultDeviceAllocator& _allocator,
                                    cudaStream_t& _cuda_stream,
                                    int32_t block_size = 32)
{
    AnchorToPosStruct anchor_to_pos_op;
    cub::TransformInputIterator<int32_t, AnchorToPosStruct, const Anchor*> anchor_q_pos(d_anchors.data(), anchor_to_pos_op);

    device_buffer<int32_t> d_query_start_runs(n_anchors, _allocator, _cuda_stream);
    device_buffer<int32_t> d_query_start_run_lengths(n_anchors, _allocator, _cuda_stream);
    device_buffer<int32_t> d_query_start_run_starts(n_anchors, _allocator, _cuda_stream);
    device_buffer<int32_t> d_num_runs(1, _allocator, _cuda_stream);

    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       anchor_q_pos,
                                       d_query_start_runs.data(),
                                       d_query_start_run_lengths.data(),
                                       d_num_runs.data(),
                                       n_anchors);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       anchor_q_pos,
                                       d_query_start_runs.data(),
                                       d_query_start_run_lengths.data(),
                                       d_num_runs.data(),
                                       n_anchors);

    std::size_t n_runs = cudautils::get_value_from_device(d_num_runs.data(), _cuda_stream);
    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  d_query_start_run_lengths.data(),
                                  d_query_start_run_starts.data(),
                                  n_runs,
                                  _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  d_query_start_run_lengths.data(),
                                  d_query_start_run_starts.data(),
                                  n_runs,
                                  _cuda_stream);

    mask_anchor_repeat_runs<<<(n_runs / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchors.data(),
                                                                                        d_mask.data(),
                                                                                        d_query_start_run_starts.data(),
                                                                                        d_query_start_run_lengths.data(),
                                                                                        n_anchors,
                                                                                        n_runs,
                                                                                        max_placements);
}

__global__ void mask_repetitive_anchors(const Anchor* anchors,
                                        bool* anchor_mask,
                                        const int32_t n_anchors,
                                        const int32_t window_size,
                                        const int32_t max_acceptable_anchor_run)
{
    // thread ID, which is used to index into the Anchors array
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_anchors)
    {
        // First index within the anchors array
        std::size_t anchor_index_start = d_tid * window_size;
        std::size_t anchor_index_end   = min(int(anchor_index_start + window_size), int(n_anchors));
        /**
        */
        int32_t max_run_length     = 0;
        int32_t current_run_length = 1;
        int32_t max_run_end        = 0;
        for (std::size_t i = anchor_index_start + 1; i < anchor_index_end; ++i)
        {
            anchor_mask[i - 1] = true;
            anchor_mask[i]     = true;
            if (anchors[i].query_position_in_read_ == anchors[i - 1].query_position_in_read_)
            {
                current_run_length += 1;
                max_run_length = max(max_run_length, current_run_length);
            }

            else
            {
                max_run_end        = current_run_length == max_run_length ? i : max_run_end;
                current_run_length = 0;
            }
        }
        if (max_run_length > max_acceptable_anchor_run)
        {
            for (int32_t i = max_run_end; i >= max_run_end - max_run_length; --i)
            {
                anchor_mask[i] = false;
            }
        }
    }
}

__global__ void initalize_anchors_mask(bool* anchors_mask, const size_t n_anchors, bool val)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_anchors)
    {
        anchors_mask[d_tid] = val;
    }
}

__global__ void
initialize_overlaps_array(Overlap* overlaps, const size_t n_overlaps)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        init_overlap(overlaps[d_tid]);
    }
}

__global__ void init_overlap_scores(const Overlap* overlaps, double* scores, const int32_t n_overlaps, const double exp)

{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        scores[d_tid] = pow(double(overlaps[d_tid].num_residues_), exp);
    }
}

__global__ void init_overlap_mask(bool* mask, const int32_t n_overlaps, const bool value)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        mask[d_tid] = value;
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
__global__ void anchmers_to_overlaps(const Anchmer* anchmers,
                                     const int32_t* overlap_ends,
                                     const size_t n_anchmers,
                                     const Anchor* anchors,
                                     const bool* anchor_mask,
                                     const size_t n_anchors,
                                     Overlap* overlaps,
                                     bool* overlap_mask,
                                     const size_t n_overlaps)
{
    // thread ID, which is used to index into the anchmers array
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (d_tid < n_anchmers)
    {
        for (std::size_t i = 0; i < anchmers[d_tid].n_anchors; ++i)
        {
            //if (anchor_mask[i])
            if (true)
            {
                std::size_t overlap_index = overlap_ends[d_tid] - anchmers[d_tid].n_chains + anchmers[d_tid].chain_id[i];
                std::size_t anchor_index  = d_tid * MAX_ANCHMER_WINDOW + i;

                add_anchor_to_overlap(anchors[anchor_index], overlaps[overlap_index]);
            }
        }
    }
}

__global__ void
generate_anchmers(const Anchor* d_anchors,
                  const bool* d_anchor_select_mask,
                  const size_t n_anchors,
                  Anchmer* anchmers,
                  const uint8_t anchmer_size)
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
    anchmers[d_tid].n_chains    = 0;
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
        std::size_t max_ind             = min(int(n_anchors - first_anchor_index), int(anchmer_size));
        if (global_anchor_index < n_anchors)
        {
            ++(anchmers[d_tid].n_anchors);
            if (!d_anchor_select_mask[global_anchor_index])
            {
                continue;
            }
            //anchmers[d_tid].n_chains = anchmers[d_tid].chain_id[i] == 0 ? anchmers[d_tid].n_chains + 1 : anchmers[d_tid].n_chains;
            //Label the anchor with its chain ID
            anchmers[d_tid].chain_id[i] = anchmers[d_tid].chain_id[i] == 0 ? ++current_chain : anchmers[d_tid].chain_id[i];
            anchmers[d_tid].n_chains    = max(anchmers[d_tid].chain_id[i], anchmers[d_tid].n_chains);
            for (std::size_t j = i + 1; j < max_ind; ++j)
            {
                if (d_anchors[global_anchor_index] == d_anchors[first_anchor_index + j])
                {
                    anchmers[d_tid].chain_id[j] = anchmers[d_tid].chain_id[i];
                }
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

__device__ __forceinline__ bool overlaps_mergable(const Overlap& a, const Overlap& b, std::int32_t max_dist, float min_gap_ratio)
{
    const int q_diff      = abs(int(b.query_start_position_in_read_) - int(a.query_end_position_in_read_));
    const int t_diff      = abs(int(a.target_end_position_in_read_) - int(b.target_start_position_in_read_));
    const bool q_adjacent = q_diff <= max_dist;
    const bool t_adjacent = t_diff <= max_dist;
    const float gap_ratio = float(min(t_diff, q_diff)) / float(max(t_diff, q_diff));
    return a.query_read_id_ == b.query_read_id_ &&
           a.target_read_id_ == b.target_read_id_ &&
           (a.relative_strand == b.relative_strand || min(a.num_residues_, b.num_residues_) == 1) &&
           ((t_adjacent && q_adjacent) || (gap_ratio > min_gap_ratio && t_diff < 5000 && q_diff < 5000));
    //(t_adjacent && q_adjacent);
}

__global__ void chain_overlaps_in_window(Overlap* overlaps,
                                         bool* overlap_mask,
                                         double* scores,
                                         const std::int32_t n_overlaps,
                                         const int32_t n_overlapmers,
                                         const std::int32_t overlapmer_size,
                                         const std::int32_t max_dist,
                                         const float max_gap_ratio)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlapmers)
    {
        std::int32_t first_overlap_index = d_tid * overlapmer_size;
        std::int32_t max_ind             = min(overlapmer_size, n_overlaps - first_overlap_index);
        for (std::size_t i = 0; i < overlapmer_size && i < n_overlaps; ++i)
        {
            std::size_t global_overlap_index = first_overlap_index + i;
            if (global_overlap_index < n_overlaps)
            {
                for (std::int32_t j = i + 1; j < max_ind; ++j)
                {
                    if (overlap_mask[first_overlap_index + j] && overlaps_mergable(overlaps[global_overlap_index], overlaps[first_overlap_index + j], max_dist, 0.8))
                    {
                        overlaps[first_overlap_index + i]     = merge_helper(overlaps[first_overlap_index + i], overlaps[first_overlap_index + j]);
                        overlap_mask[first_overlap_index + j] = false;
                        scores[first_overlap_index + i]       = scores[first_overlap_index + i] + scores[first_overlap_index + j];
                    }
                }
            }
        }
    }
}

__global__ void init_predecessor_and_score_arrays(int32_t* predecessors,
                                                  double* scores,
                                                  bool* mask,
                                                  int32_t n_overlaps)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        predecessors[d_tid] = d_tid;
    }
}

__device__ __forceinline__ int32_t exp_gap_cost(Overlap& a, Overlap& b)
{

    int32_t id_match_cost               = a.query_read_id_ == b.query_read_id_ && a.target_read_id_ == b.target_read_id_ ? 0 : INT32_INFINITY;
    int32_t q_diff                      = abs(int(b.query_start_position_in_read_) - int(a.query_end_position_in_read_));
    const bool relative_strands_reverse = (a.relative_strand == claraparabricks::genomeworks::cudamapper::RelativeStrand::Reverse) && (b.relative_strand == claraparabricks::genomeworks::cudamapper::RelativeStrand::Reverse);

    int32_t t_diff      = relative_strands_reverse ? abs(int(a.target_start_position_in_read_) - int(b.target_end_position_in_read_)) : abs(int(b.target_start_position_in_read_) - int(a.target_end_position_in_read_));
    int32_t strand_cost = a.relative_strand == b.relative_strand || a.num_residues_ == 1 || b.num_residues_ == 1 ? 0 : INT32_INFINITY;

    float gap_ratio = min(float(q_diff), float(t_diff)) / max(float(q_diff), float(t_diff));

    int32_t ratio_cost = min(int(100), int(1.0 / pow(gap_ratio, 5)));
    int32_t diff_cost  = max(int(q_diff), int(t_diff)) < 5000 ? 0 : 100;
    return ratio_cost + diff_cost + strand_cost + id_match_cost;
}

__global__ void chain_overlaps_by_score(Overlap* overlaps,
                                        double* scores,
                                        bool* select_mask,
                                        int32_t* predecessors,
                                        const int32_t n_overlaps,
                                        const int32_t max_distance,
                                        const int32_t max_iter)
{
    // Expects one thread per overlap
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        int32_t global_overlap_index = d_tid;
        int32_t end_index            = min(int(global_overlap_index + max_iter), int(n_overlaps));
        for (int32_t i = global_overlap_index; i < end_index; ++i)
        {
            int32_t i_score = scores[i];
            for (int32_t j = i + 1; j < i + 20; ++j)
            {
                int32_t score_addition  = min(int(scores[j]), int(100));
                int32_t gap_cost        = exp_gap_cost(overlaps[i], overlaps[j]);
                int32_t marginal_score  = min(int(score_addition), int(100)) - int(gap_cost);
                int32_t tentative_score = min(i_score + marginal_score, 100);
                if (tentative_score > scores[j])
                {
                    scores[j]       = i_score + score_addition - gap_cost;
                    predecessors[j] = i;
                    select_mask[i]  = false;
                }
            }
        }
    }
}

__global__ void produce_chains(Overlap* overlaps,
                               double* scores,
                               int32_t* predecessors,
                               bool* select_mask,
                               int32_t n_overlaps,
                               double min_score)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        int32_t global_overlap_index = d_tid;

        if (true)
        {
            int32_t index = global_overlap_index;
            Overlap final_overlap;
            final_overlap      = overlaps[index];
            double final_score = scores[index];
            while (index != predecessors[index])
            {
                int32_t pred      = predecessors[index];
                final_overlap     = merge_helper(final_overlap, overlaps[pred]);
                select_mask[pred] = false;
                index             = pred;
            }
            overlaps[global_overlap_index] = final_overlap;
            //select_mask[global_overlap_index] = true;
            scores[global_overlap_index] = final_score;
            // printf("Chain start: %d %d %d %d %d %d %d %f\n",
            //        final_overlap.query_read_id_,
            //        final_overlap.query_start_position_in_read_,
            //        final_overlap.query_end_position_in_read_,
            //        final_overlap.target_read_id_,
            //        final_overlap.target_start_position_in_read_,
            //        final_overlap.target_end_position_in_read_,
            //        final_overlap.num_residues_,
            //        final_score);
        }
    }
}

__global__ void flip_adjacent_sign(Overlap* overlaps, const int32_t n_overlaps)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid > 0 && d_tid < n_overlaps - 1 && overlaps[d_tid].num_residues_ == 1 && overlaps[d_tid - 1].num_residues_ > 1)
    {
        RelativeStrand left_strand      = overlaps[d_tid - 1].relative_strand;
        overlaps[d_tid].relative_strand = left_strand;
    }
    else if (d_tid > 0 && d_tid < n_overlaps - 1 && overlaps[d_tid].num_residues_ == 1 && overlaps[d_tid + 1].num_residues_ > 1)
    {
        RelativeStrand right_strand     = overlaps[d_tid + 1].relative_strand;
        overlaps[d_tid].relative_strand = right_strand;
    }
}

__global__ void drop_single_anchor_overlaps(Overlap* overlaps, bool* mask, const int32_t num_overlaps)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < num_overlaps)
    {
        mask[d_tid] = overlaps[d_tid].query_start_position_in_read_ != overlaps[d_tid].query_end_position_in_read_ && overlaps[d_tid].target_start_position_in_read_ != overlaps[d_tid].target_end_position_in_read_;
    }
}

/**
* Mark primary chains with a score of 1 and secondaries with a score of 0.
* TODO: implement secondary chaining check.
*/
__global__ void primary_chains_in_query_target_pairs(Overlap* overlaps,
                                                     bool* select_mask,
                                                     double* scores,
                                                     const int32_t* qt_starts,
                                                     const int32_t* qt_ends,
                                                     const int32_t num_overlaps,
                                                     const int32_t num_qt_chains,
                                                     const int32_t max_lookback,
                                                     const int32_t minimum_chain_score,
                                                     const int32_t max_dist)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < num_qt_chains)
    {
        //int16_t current_chain = 1;
        const int32_t start = qt_starts[d_tid];
        const int32_t end   = qt_ends[d_tid];
        if (start != end)
        {

            // for (int32_t i = start + 1; i < end; ++i)
            // {
            //     const bool strand_equal = overlaps[i].relative_strand == overlaps[i - 1].relative_strand;
            //     double tmp_score        = scores[i] + scores[i - 1];
            //     const int32_t q_gap     = overlaps[i].query_start_position_in_read_ - overlaps[i - 1].query_end_position_in_read_;
            //     const int32_t t_gap     = overlaps[i].relative_strand == RelativeStrand::Forward ? overlaps[i].target_start_position_in_read_ - overlaps[i - 1].target_end_position_in_read_ : overlaps[i].target_end_position_in_read_ - overlaps[i - 1].target_start_position_in_read_;
            //     const int32_t gap_cost  = max(q_gap, t_gap);
            //     if (strand_equal && gap_cost < max_dist && (min(int(tmp_score), int(100)) - (int(gap_cost) / 100)) > 50)
            //     {
            //         overlaps[i]        = merge_helper(overlaps[i - 1], overlaps[i]);
            //         select_mask[i - 1] = false;
            //         scores[i]          = tmp_score;
            //     }
            // }

            for (int32_t i = start; i < end; ++i)
            {
                if (select_mask[i])
                {
                    for (int32_t j = start; j < i; ++j)
                    {
                        const bool strand_equal = overlaps[i].relative_strand == overlaps[j].relative_strand;
                        double tmp_score        = scores[i] + scores[j];
                        const int32_t q_gap     = overlaps[i].query_start_position_in_read_ - overlaps[j].query_end_position_in_read_;
                        const int32_t t_gap     = overlaps[i].relative_strand == RelativeStrand::Forward ? overlaps[i].target_start_position_in_read_ - overlaps[j].target_end_position_in_read_ : overlaps[i].target_end_position_in_read_ - overlaps[j].target_start_position_in_read_;
                        const int32_t gap_cost  = max(q_gap, t_gap);
                        if (select_mask[j] && gap_cost < max_dist && strand_equal && (min(int(tmp_score), int(100)) - (int(gap_cost) / 100)) > minimum_chain_score)
                        {
                            overlaps[i]    = merge_helper(overlaps[i], overlaps[j]);
                            select_mask[j] = false;
                            scores[j]      = tmp_score;
                        }
                    }
                }
            }

            for (int32_t i = start; i < end; ++i)
            {
                select_mask[i] &= scores[i] > minimum_chain_score;
            }
        }
    }
}

void drop_scores_by_mask(device_buffer<double>& d_scores,
                         device_buffer<bool>& d_mask,
                         const std::int32_t n_overlaps,
                         device_buffer<double>& d_dest,
                         device_buffer<int32_t>& d_filtered_count,
                         DefaultDeviceAllocator& _allocator,
                         cudaStream_t& _cuda_stream)
{
    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage,
                               temp_storage_bytes,
                               d_scores.data(),
                               d_mask.data(),
                               d_dest.data(),
                               d_filtered_count.data(),
                               n_overlaps,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage,
                               temp_storage_bytes,
                               d_scores.data(),
                               d_mask.data(),
                               d_dest.data(),
                               d_filtered_count.data(),
                               n_overlaps,
                               _cuda_stream);
}

void drop_overlaps_by_mask(device_buffer<Overlap>& d_overlaps,
                           device_buffer<bool>& d_mask,
                           const std::int32_t n_overlaps,
                           device_buffer<Overlap>& d_dest,
                           device_buffer<int32_t>& d_filtered_count,
                           DefaultDeviceAllocator& _allocator,
                           cudaStream_t& _cuda_stream)
{
    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_overlaps.data(),
                               d_mask.data(),
                               d_dest.data(),
                               d_filtered_count.data(),
                               n_overlaps,
                               _cuda_stream);
    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_overlaps.data(),
                               d_mask.data(),
                               d_dest.data(),
                               d_filtered_count.data(),
                               n_overlaps,
                               _cuda_stream);
}

void encode_query_target_pairs(Overlap* overlaps,
                               int32_t n_overlaps,
                               device_buffer<int32_t>& query_target_starts,
                               device_buffer<int32_t>& query_target_lengths,
                               device_buffer<int32_t>& query_target_ends,
                               int32_t& n_query_target_pairs,
                               DefaultDeviceAllocator& _allocator,
                               cudaStream_t& _cuda_stream,
                               int32_t block_size = 32)
{
    OverlapToQueryTargetPairOp qt_pair_op;
    cub::TransformInputIterator<QueryTargetPair, OverlapToQueryTargetPairOp, Overlap*> d_query_target_pairs(overlaps, qt_pair_op);
    device_buffer<QueryTargetPair> d_qt_pairs(n_overlaps, _allocator, _cuda_stream);
    device_buffer<int32_t> d_num_query_target_pairs(1, _allocator, _cuda_stream);

    device_buffer<char> d_temp_buf(_allocator, _cuda_stream);
    void* d_temp_storage           = nullptr;
    std::size_t temp_storage_bytes = 0;

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_overlaps);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                       temp_storage_bytes,
                                       d_query_target_pairs,
                                       d_qt_pairs.data(),
                                       query_target_lengths.data(),
                                       d_num_query_target_pairs.data(),
                                       n_overlaps);

    n_query_target_pairs = cudautils::get_value_from_device(d_num_query_target_pairs.data(), _cuda_stream);

    d_temp_storage     = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs, _cuda_stream);

    d_temp_buf.clear_and_resize(temp_storage_bytes);
    d_temp_storage = d_temp_buf.data();

    cub::DeviceScan::ExclusiveSum(d_temp_storage,
                                  temp_storage_bytes,
                                  query_target_lengths.data(),
                                  query_target_starts.data(),
                                  n_query_target_pairs, _cuda_stream);

    convert_offsets_to_ends<<<(n_query_target_pairs / block_size) + 1, block_size, 0, _cuda_stream>>>(query_target_starts.data(), query_target_lengths.data(), query_target_ends.data(), n_query_target_pairs);
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
    const std::int32_t anchors_per_anchmer    = MAX_ANCHMER_WINDOW;
    const std::int32_t overlapmer_window_size = MAX_OVERLAPMER_WINDOW;
    const std::size_t n_anchors               = d_anchors.size();
    const std::size_t n_anchmers              = (d_anchors.size() / anchors_per_anchmer) + 1;
    const std::int32_t block_size             = 32;

    //const std::int32_t max_gap = 5000;

    //std::vector<Anchmer> anchmers(n_anchmers);
    device_buffer<Anchmer> d_anchmers(n_anchmers, _allocator, _cuda_stream);
    device_buffer<bool> d_anchor_select_mask(n_anchors, _allocator, _cuda_stream);

    initalize_anchors_mask<<<(n_anchors / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchor_select_mask.data(),
                                                                                          n_anchors,
                                                                                          true);

    //mask_repetitive_anchors<<<(n_anchors / 20) + 1, block_size, 0, _cuda_stream>>>(d_anchors.data(), d_anchor_select_mask.data(), n_anchors, 12, 5);
    //mask_repetitive_anchors_by_RLE(d_anchors, d_anchor_select_mask, n_anchors, 5, _allocator, _cuda_stream, block_size);
    // Stage one: generate anchmers
    generate_anchmers<<<(n_anchmers / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchors.data(),
                                                                                      d_anchor_select_mask.data(),
                                                                                      n_anchors,
                                                                                      d_anchmers.data(),
                                                                                      anchors_per_anchmer);

    // #define DEBUG
    // #ifdef DEBUG
    //     std::vector<Anchor> anchors;
    //     //std::vector<bool> host_masks;
    //     //host_masks.resize(n_anchors);
    //     anchors.resize(n_anchors);
    //     cudautils::device_copy_n(d_anchors.data(), n_anchors, anchors.data(), _cuda_stream);
    //     //cudaMemcpy(masks.data(), d_anchor_select_mask.data(), n_anchors * sizeof(bool), cudaMemcpyDefault);
    //     //cudautils::device_copy_n(d_anchor_select_mask.data(), n_anchors, host_masks.data(), _cuda_stream);
    //     for (std::size_t i = 0; i < n_anchors; ++i)
    //     {
    //         Anchor a = anchors[i];
    //         std::cerr << a.query_position_in_read_;
    //         std::cerr << " " << a.target_position_in_read_ << std::endl;
    //         // std::cerr << (host_masks[i] ? "true" : "false") << std::endl;
    //     }
    // #endif
    // #ifdef DEBUG
    //     std::vector<Anchmer> anchmers;
    //     std::vector<Anchor> anchors;
    //     anchmers.resize(d_anchmers.size());
    //     anchors.resize(n_anchors);
    //     cudautils::device_copy_n(d_anchmers.data(), d_anchmers.size(), anchmers.data(), _cuda_stream);
    //     cudautils::device_copy_n(d_anchors.data(), n_anchors, anchors.data(), _cuda_stream);
    //     for (std::size_t i = 0; i < anchmers.size(); ++i)
    //     {
    //         Anchmer a = anchmers[i];
    //         std::cerr << a.n_anchors << " " << static_cast<int16_t>(a.n_chains) << std::endl;
    //         for (std::size_t j = 0; j < a.n_anchors; ++j)
    //         {
    //             std::cerr << static_cast<int16_t>(a.chain_id[j]) << " ";
    //         }
    //         std::cerr << std::endl;
    //         for (std::size_t j = 0; j < a.n_anchors; ++j)
    //         {
    //             std::cerr << static_cast<int16_t>(anchors[(MAX_ANCHMER_WINDOW * i) + j].query_position_in_read_) << " ";
    //         }

    //         std::cerr << std::endl;
    //     }
    // #endif

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

    //std::cerr << "Generating " << n_initial_overlaps << " initial overlaps from " << n_anchmers << " anchmers..." << std::endl;

    // Create device buffers needed to hold all future values.
    // Overlaps_SRC and Overlaps_DEST provide two vectors for overlaps so that filtering can be done in rounds.
    // Overlaps_SELECT_MASK provides a single boolean mask for all overlaps
    // query_target_starts provides the start indicies of the query-target pairs in the overlaps vector.
    device_buffer<Overlap> d_overlaps_source(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<Overlap> d_overlaps_dest(n_initial_overlaps, _allocator, _cuda_stream);

    device_buffer<bool> d_overlaps_select_mask(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<bool> d_chain_max(n_initial_overlaps, _allocator, _cuda_stream);

    device_buffer<double> d_overlap_scores(n_initial_overlaps, _allocator, _cuda_stream);
    device_buffer<double> d_overlap_scores_dest(n_initial_overlaps, _allocator, _cuda_stream);

    device_buffer<int32_t> d_predecessors(n_initial_overlaps, _allocator, _cuda_stream);

    initialize_overlaps_array<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
                                                                                                      n_initial_overlaps);
    init_overlap_mask<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_select_mask.data(),
                                                                                              n_initial_overlaps,
                                                                                              true);

    anchmers_to_overlaps<<<(n_anchmers / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchmers.data(),
                                                                                         d_overlap_ends.data(),
                                                                                         n_anchmers,
                                                                                         d_anchors.data(),
                                                                                         d_anchor_select_mask.data(),
                                                                                         n_anchors,
                                                                                         d_overlaps_source.data(),
                                                                                         d_overlaps_select_mask.data(),
                                                                                         n_initial_overlaps);
    flip_adjacent_sign<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
                                                                                               n_initial_overlaps);

    init_overlap_scores<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
                                                                                                d_overlap_scores.data(),
                                                                                                n_initial_overlaps,
                                                                                                2.0);

    d_anchmers.clear_and_resize(0);
    d_anchor_select_mask.clear_and_resize(0);

//#define DEBUG
#ifdef DEBUG
    // std::size_t num_overlaps = n_initial_overlaps;
    // std::vector<Overlap>
    //     intermediate_overlaps;
    // std::vector<double> intermediate_scores;
    // intermediate_scores.resize(num_overlaps);
    // cudautils::device_copy_n(d_overlap_scores.data(), num_overlaps, intermediate_scores.data(), _cuda_stream);
    // intermediate_overlaps.resize(num_overlaps);
    // cudautils::device_copy_n(d_overlaps_source.data(), num_overlaps, intermediate_overlaps.data(), _cuda_stream);
    // for (std::size_t i = 0; i < intermediate_overlaps.size(); ++i)
    // {
    //     Overlap o    = intermediate_overlaps[i];
    //     double score = intermediate_scores[i];
    //     std::cerr << o.query_read_id_ << " " << o.query_start_position_in_read_;
    //     std::cerr << " " << o.query_end_position_in_read_ << " ";
    //     std::cerr << static_cast<char>(o.relative_strand) << " " << char(o.relative_strand);
    //     std::cerr << " " << o.target_read_id_ << " " << o.target_start_position_in_read_ << " ";
    //     std::cerr << o.target_end_position_in_read_ << " " << o.num_residues_ << " " << score << std::endl;
    // }
    // std::cerr << std::endl
    //           << std::endl;
#endif

    // chain_overlaps_in_window<<<(n_initial_overlaps / overlapmer_window_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
    //                                                                                                              d_overlaps_select_mask.data(),
    //                                                                                                              d_overlap_scores.data(),
    //                                                                                                              n_initial_overlaps, (n_initial_overlaps / overlapmer_window_size) + 1,
    //                                                                                                              overlapmer_window_size,
    //                                                                                                              500,
    //                                                                                                              0.8);

    init_predecessor_and_score_arrays<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_predecessors.data(), d_overlap_scores.data(), d_overlaps_select_mask.data(), n_initial_overlaps);

    init_overlap_mask<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_select_mask.data(),
                                                                                              n_initial_overlaps,
                                                                                              true);

    chain_overlaps_by_score<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
                                                                                                    d_overlap_scores.data(),
                                                                                                    d_overlaps_select_mask.data(),
                                                                                                    d_predecessors.data(), n_initial_overlaps,
                                                                                                    5000,
                                                                                                    32);

    produce_chains<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
                                                                                           d_overlap_scores.data(),
                                                                                           d_predecessors.data(),
                                                                                           d_overlaps_select_mask.data(),
                                                                                           n_initial_overlaps,
                                                                                           50);

    device_buffer<int32_t> d_n_filtered_overlaps(1, _allocator, _cuda_stream);

    // device_buffer<int32_t> d_query_target_pair_starts(n_initial_overlaps, _allocator, _cuda_stream);
    // device_buffer<int32_t> d_query_target_pair_lengths(n_initial_overlaps, _allocator, _cuda_stream);
    // device_buffer<int32_t> d_query_target_pair_ends(n_initial_overlaps, _allocator, _cuda_stream);
    // int32_t n_query_target_pairs = 0;

    // encode_query_target_pairs(d_overlaps_source.data(),
    //                           n_initial_overlaps,
    //                           d_query_target_pair_starts,
    //                           d_query_target_pair_lengths,
    //                           d_query_target_pair_ends,
    //                           n_query_target_pairs,
    //                           _allocator,
    //                           _cuda_stream,
    //                           block_size);

#ifdef DEBUG
    // num_overlaps = n_initial_overlaps;
    // intermediate_scores.resize(num_overlaps);
    // cudautils::device_copy_n(d_overlap_scores.data(), num_overlaps, intermediate_scores.data(), _cuda_stream);

    // intermediate_overlaps.resize(num_overlaps);
    // cudautils::device_copy_n(d_overlaps_source.data(), num_overlaps, intermediate_overlaps.data(), _cuda_stream);

    // std::vector<int32_t> intermediate_preds;
    // intermediate_preds.resize(num_overlaps);
    // cudautils::device_copy_n(d_predecessors.data(), num_overlaps, intermediate_preds.data(), _cuda_stream);
    // for (std::size_t i = 0; i < intermediate_overlaps.size(); ++i)
    // {
    //     Overlap o    = intermediate_overlaps[i];
    //     double score = intermediate_scores[i];
    //     int32_t pred = intermediate_preds[i];
    //     std::cerr << o.query_read_id_ << " " << o.query_start_position_in_read_;
    //     std::cerr << " " << o.query_end_position_in_read_ << " ";
    //     std::cerr << static_cast<char>(o.relative_strand) << " ";
    //     std::cerr << " " << o.target_read_id_ << " " << o.target_start_position_in_read_ << " ";
    //     // std::cerr << o.target_end_position_in_read_ << " " << o.num_residues_ << std::endl;
    //     std::cerr << o.target_end_position_in_read_ << " " << o.num_residues_;
    //     std::cerr << " " << score << " " << i << " " << pred << std::endl;
    // }
#endif

    // primary_chains_in_query_target_pairs<<<(n_query_target_pairs / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
    //                                                                                                                d_overlaps_select_mask.data(),
    //                                                                                                                d_overlap_scores.data(),
    //                                                                                                                d_query_target_pair_starts.data(),
    //                                                                                                                d_query_target_pair_ends.data(),
    //                                                                                                                n_initial_overlaps,
    //                                                                                                                n_query_target_pairs,
    //                                                                                                                0,
    //                                                                                                                20,
    //                                                                                                                5000);

    mask_overlaps<<<(n_initial_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
                                                                                          n_initial_overlaps,
                                                                                          d_overlaps_select_mask.data(),
                                                                                          min_overlap_len,
                                                                                          min_residues,
                                                                                          min_bases_per_residue,
                                                                                          all_to_all,
                                                                                          true);

    drop_overlaps_by_mask(d_overlaps_source,
                          d_overlaps_select_mask,
                          n_initial_overlaps,
                          d_overlaps_dest,
                          d_n_filtered_overlaps,
                          _allocator,
                          _cuda_stream);
    drop_scores_by_mask(d_overlap_scores,
                        d_overlaps_select_mask,
                        n_initial_overlaps,
                        d_overlap_scores_dest,
                        d_n_filtered_overlaps,
                        _allocator,
                        _cuda_stream);
    int32_t n_filtered_overlaps = cudautils::get_value_from_device(d_n_filtered_overlaps.data(), _cuda_stream);

    fused_overlaps.resize(n_filtered_overlaps);
    cudautils::device_copy_n(d_overlaps_dest.data(), n_filtered_overlaps, fused_overlaps.data(), _cuda_stream);

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
