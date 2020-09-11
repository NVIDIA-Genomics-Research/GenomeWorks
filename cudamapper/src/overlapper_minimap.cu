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

#include "overlapper_minimap.hpp"

#include <fstream>
#include <sstream>
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

//#define CHAINDEBUG

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

#define INT32_INFINITY 100000000
#define NEGATIVE_INT32_INFINITY -1 * INT32_INFINITY

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

__device__ __forceinline__ double percent_reciprocal_overlap(const Overlap& a, const Overlap& b)
{
    if (a.query_read_id_ != b.query_read_id_ || a.target_read_id_ != b.target_read_id_ || a.relative_strand != b.relative_strand)
    {
        return 0.0;
    }
    int32_t query_overlap = min(a.query_end_position_in_read_, b.query_end_position_in_read_) - max(a.query_start_position_in_read_, b.query_start_position_in_read_);
    int32_t target_overlap;
    if (a.relative_strand == RelativeStrand::Forward && b.relative_strand == RelativeStrand::Forward)
    {
        target_overlap = min(a.target_end_position_in_read_, b.target_end_position_in_read_) - max(a.target_start_position_in_read_, b.target_start_position_in_read_);
    }
    else
    {
        target_overlap = max(a.target_start_position_in_read_, b.target_start_position_in_read_) - min(a.target_end_position_in_read_, b.target_end_position_in_read_);
    }

    int32_t query_total_length = max(a.query_end_position_in_read_, b.query_end_position_in_read_) - min(a.query_start_position_in_read_, b.query_start_position_in_read_);
    int32_t target_total_length;
    if (a.relative_strand == RelativeStrand::Forward && b.relative_strand == RelativeStrand::Forward)
    {
        target_total_length = max(a.target_end_position_in_read_, b.target_end_position_in_read_) - min(a.target_start_position_in_read_, b.target_start_position_in_read_);
    }
    else
    {
        target_total_length = min(a.target_start_position_in_read_, b.target_start_position_in_read_) - max(a.target_end_position_in_read_, b.target_end_position_in_read_);
    }
    return static_cast<double>(query_overlap + target_overlap) / static_cast<double>(query_total_length + target_total_length);
}

// Checks if Overlap a is contained within Overlap b.
__device__ __forceinline__ bool contained_overlap(const Overlap& a, const Overlap& b)
{
    if (a.query_read_id_ != b.query_read_id_ || a.target_read_id_ != b.target_read_id_ || a.relative_strand != b.relative_strand)
        return false;
    bool query_contained = a.query_start_position_in_read_ >= b.query_start_position_in_read_ && a.query_end_position_in_read_ <= b.query_end_position_in_read_;
    bool target_contained;
    if (a.relative_strand == RelativeStrand::Forward)
    {
        target_contained = a.target_start_position_in_read_ >= b.target_start_position_in_read_ && a.target_end_position_in_read_ <= b.target_end_position_in_read_;
    }
    else
    {
        target_contained = a.target_end_position_in_read_ >= b.target_end_position_in_read_ && a.target_start_position_in_read_ <= b.target_end_position_in_read_;
    }

    return query_contained && target_contained;
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

__global__ void mask_overlaps(Overlap* overlaps, std::size_t n_overlaps, bool* select_mask,
                              const std::size_t min_overlap_length,
                              const std::size_t min_residues,
                              const std::size_t max_bases_per_residue,
                              const bool all_to_all,
                              const bool filter_self_mappings,
                              const double max_percent_reciprocal,
                              const int32_t max_reciprocal_iterations)
{
    std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        position_in_read_t overlap_query_length  = overlaps[d_tid].query_end_position_in_read_ - overlaps[d_tid].query_start_position_in_read_;
        position_in_read_t overlap_target_length = overlaps[d_tid].target_end_position_in_read_ - overlaps[d_tid].target_start_position_in_read_;
        //const bool mask_self_self                = overlaps[d_tid].query_read_id_ == overlaps[d_tid].target_read_id_ && all_to_all && filter_self_mappings;
        const bool mask_self_self     = false;
        auto query_bases_per_residue  = static_cast<double>(overlap_query_length) / static_cast<double>(overlaps[d_tid].num_residues_);
        auto target_bases_per_residue = static_cast<double>(overlap_target_length) / static_cast<double>(overlaps[d_tid].num_residues_);
        select_mask[d_tid] &= overlap_query_length >= min_overlap_length & overlap_target_length >= min_overlap_length;
        select_mask[d_tid] &= overlaps[d_tid].num_residues_ >= min_residues;
        //mask[d_tid] &= !mask_self_self;
        select_mask[d_tid] &= (query_bases_per_residue < max_bases_per_residue && target_bases_per_residue < max_bases_per_residue);
        for (int32_t i = d_tid + 1; i < d_tid + max_reciprocal_iterations; ++i)
        {
            if (i < n_overlaps)
            {
                if (percent_reciprocal_overlap(overlaps[d_tid], overlaps[i]) > max_percent_reciprocal || contained_overlap(overlaps[d_tid], overlaps[i]))
                    select_mask[d_tid] = false;
            }
        }
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

    if (c.target_start_position_in_read_ > c.target_end_position_in_read_)
    {
        c.relative_strand = RelativeStrand::Reverse;
    }
    return c;
}

__device__ __forceinline__ void init_overlap(Overlap& overlap)
{
    overlap.query_read_id_                 = 0;
    overlap.target_read_id_                = 0;
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

__global__ void initalize_anchors_mask(bool* anchors_mask, const size_t n_anchors, bool val)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_anchors)
    {
        anchors_mask[d_tid] = val;
    }
}

__global__ void initialize_overlaps_array(Overlap* overlaps, const size_t n_overlaps)
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

__global__ void init_overlap_scores_to_value(double* scores, double val, const int32_t n_overlaps)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        scores[d_tid] = 0;
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

__global__ void init_predecessor_and_score_arrays(int32_t* predecessors,
                                                  double* scores,
                                                  bool* mask,
                                                  int32_t n_overlaps)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        scores[d_tid]       = 0;
        predecessors[d_tid] = d_tid;
    }
}

__device__ __forceinline__ int32_t fast_approx_log2(const int32_t val)
{
    if (val < 2)
        return 0;
    else if (val < 4)
        return 1;
    else if (val < 8)
        return 2;
    else if (val < 16)
        return 3;
    else if (val < 32)
        return 4;
    else if (val < 64)
        return 5;
    else if (val < 128)
        return 6;
    else if (val < 256)
        return 7;
    else
        return 8;
}

__device__ __forceinline__ int32_t log_linear_anchor_weight(const Anchor& a,
                                                            const Anchor& b,
                                                            const int32_t max_dist,
                                                            const int32_t max_bandwidth)
{
    int32_t word_size = 15;
    if (a.query_read_id_ != b.query_read_id_ || a.target_read_id_ != b.target_read_id_)
        return NEGATIVE_INT32_INFINITY;

    int32_t b_query_pos  = b.query_position_in_read_ + word_size;
    int32_t b_target_pos = b.target_position_in_read_ + word_size;

    int32_t x_dist = abs(int(b_target_pos) - int(a.target_position_in_read_));

    if (x_dist > max_dist || x_dist == 0)
        return NEGATIVE_INT32_INFINITY;

    int32_t y_dist = (b_query_pos) - (a.query_position_in_read_);

    if (y_dist > max_dist || y_dist <= 0)
        return NEGATIVE_INT32_INFINITY;

    int32_t dist_diff = x_dist > y_dist ? x_dist - y_dist : y_dist - x_dist;
    if (dist_diff > max_bandwidth)
        return NEGATIVE_INT32_INFINITY;

    int32_t min_dist      = min(x_dist, y_dist);
    int32_t log_dist_diff = fast_approx_log2(dist_diff);

    int32_t min_size = word_size;
    int32_t score    = min_dist > min_size ? min_size : min_dist;
    //int32_t score = min_dist;
    score -= (double(score) * (0.01 * word_size) + double(log_dist_diff) * 0.5);
    //printf("%d %d %d %d | %d \n", x_dist, y_dist, min_dist, min_size, score);
    return score;
}

__global__ void chain_anchors_by_score(const Anchor* anchors,
                                       double* scores,
                                       int32_t* predecessors,
                                       bool* select_mask,
                                       const int32_t num_anchors,
                                       const int32_t max_distance,
                                       const int32_t max_bandwidth,
                                       const int32_t max_iter)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (d_tid < num_anchors)
    {
        const int32_t global_overlap_index = static_cast<int32_t>(d_tid);
        int32_t end_index                  = min(int(global_overlap_index + max_iter), int(num_anchors));
        int32_t i_score                    = scores[global_overlap_index];
        for (int32_t j = global_overlap_index + 1; j < end_index; ++j)
        {
            int32_t marginal_score = log_linear_anchor_weight(anchors[global_overlap_index], anchors[j], max_distance, max_bandwidth);
            int32_t temp_score     = i_score + marginal_score;
            if (temp_score > scores[j])
            {
                scores[j]                         = temp_score;
                predecessors[j]                   = global_overlap_index;
                select_mask[global_overlap_index] = false;
            }
        }
        // printf("| %d %d %d %d | %s\n",
        //        anchors[d_tid].query_read_id_,
        //        anchors[d_tid].query_position_in_read_,
        //        anchors[d_tid].target_read_id_,
        //        anchors[d_tid].target_position_in_read_,
        //        select_mask[d_tid] ? "max" : "_within_chain_");
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

__global__ void produce_anchor_chains(const Anchor* anchors,
                                      Overlap* overlaps,
                                      double* scores,
                                      bool* select_mask,
                                      int32_t* predecessors,
                                      const int32_t n_anchors,
                                      const int32_t min_score)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_anchors)
    {
        if (select_mask[d_tid])
        {
            int32_t global_overlap_index = d_tid;
            int32_t index                = global_overlap_index;
            Overlap final_overlap        = overlaps[global_overlap_index];
            double final_score           = scores[global_overlap_index];
            init_overlap(final_overlap);
            add_anchor_to_overlap(anchors[global_overlap_index], final_overlap);
            while (index != predecessors[index])
            {
                int32_t pred = predecessors[index];
                add_anchor_to_overlap(anchors[pred], final_overlap);
                select_mask[pred] = false;
                index             = predecessors[index];
            }
            overlaps[global_overlap_index] = final_overlap;
            scores[global_overlap_index]   = final_score;
            // printf("Final chain: %d %d %d | %d %d %d | %d | %f | %d %d %d %d \n",
            //        final_overlap.query_read_id_,
            //        final_overlap.query_start_position_in_read_,
            //        final_overlap.query_end_position_in_read_,
            //        final_overlap.target_read_id_,
            //        final_overlap.target_start_position_in_read_,
            //        final_overlap.target_end_position_in_read_,
            //        final_overlap.num_residues_,
            //        final_score,
            //        anchors[d_tid].query_read_id_,
            //        anchors[d_tid].query_position_in_read_,
            //        anchors[d_tid].target_read_id_,
            //        anchors[d_tid].target_position_in_read_);
        }
    }
}

__device__ __forceinline__ bool check_query_target_pair(const Overlap& a, const Overlap& b)
{
    return a.query_read_id_ == b.query_read_id_ && a.target_read_id_ == b.target_read_id_;
}

__device__ __forceinline__ double calculate_interval_overlap(const int32_t interval_start, const int32_t interval_end, const int32_t query_start, const int32_t query_end)
{
    if (query_start > interval_end || query_end < interval_start)
        return 0;
    double overlap_start   = max(double(interval_start), double(query_start));
    double overlap_end     = min(double(interval_end), double(query_end));
    double overlap         = overlap_end - overlap_start;
    double interval_length = double(interval_end) - double(interval_start);
    return overlap / interval_length;
}

__device__ __forceinline__ bool overlap_is_secondary(const Overlap& a, const Overlap& query_overlap, const double min_overlap)
{
    const double target_overlap_frac = calculate_interval_overlap(a.query_start_position_in_read_, a.query_end_position_in_read_, query_overlap.query_start_position_in_read_, query_overlap.query_end_position_in_read_);
    const double query_overlap_frac  = calculate_interval_overlap(a.target_start_position_in_read_, a.target_end_position_in_read_, query_overlap.target_start_position_in_read_, query_overlap.target_end_position_in_read_);

#ifdef CHAINDEBUG
    printf("Overlap secondary? : %d %d %d %d %d %d %d | %d %d %d %d %d %d %d : %f %f\n",
           a.query_read_id_,
           a.query_start_position_in_read_,
           a.query_end_position_in_read_,
           a.target_read_id_,
           a.target_start_position_in_read_,
           a.target_end_position_in_read_,
           a.num_residues_,

           query_overlap.query_read_id_,
           query_overlap.query_start_position_in_read_,
           query_overlap.query_end_position_in_read_,
           query_overlap.target_read_id_,
           query_overlap.target_start_position_in_read_,
           query_overlap.target_end_position_in_read_,
           query_overlap.num_residues_, query_overlap_frac, target_overlap_frac);
#endif

    return a.query_read_id_ == query_overlap.query_read_id_ &&
               a.target_read_id_ == query_overlap.target_read_id_ &&
               a.relative_strand == query_overlap.relative_strand &&
               target_overlap_frac > min_overlap ||
           query_overlap_frac > min_overlap;
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

void OverlapperMinimap::get_overlaps(std::vector<Overlap>& fused_overlaps,
                                     const device_buffer<Anchor>& d_anchors,
                                     bool all_to_all,
                                     int64_t min_residues,
                                     int64_t min_overlap_len,
                                     int64_t min_bases_per_residue,
                                     float min_overlap_fraction)
{
    const std::int32_t block_size = 32;
    const std::size_t n_anchors   = d_anchors.size();
    device_buffer<bool> d_overlaps_select_mask(n_anchors, _allocator, _cuda_stream);
    device_buffer<Overlap> d_overlaps_source(n_anchors, _allocator, _cuda_stream);
    device_buffer<Overlap> d_overlaps_dest(n_anchors, _allocator, _cuda_stream);

    device_buffer<int32_t> d_anchor_predecessors(n_anchors, _allocator, _cuda_stream);
    device_buffer<double> d_anchor_scores(n_anchors, _allocator, _cuda_stream);

    init_overlap_mask<<<(n_anchors / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_select_mask.data(),
                                                                                     n_anchors,
                                                                                     true);

    init_overlap_scores_to_value<<<(n_anchors / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchor_scores.data(), 1.0, n_anchors);

    init_predecessor_and_score_arrays<<<(n_anchors / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchor_predecessors.data(),
                                                                                                     d_anchor_scores.data(),
                                                                                                     d_overlaps_select_mask.data(),
                                                                                                     n_anchors);

    chain_anchors_by_score<<<(n_anchors / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchors.data(),
                                                                                          d_anchor_scores.data(),
                                                                                          d_anchor_predecessors.data(),
                                                                                          d_overlaps_select_mask.data(),
                                                                                          n_anchors,
                                                                                          5000,
                                                                                          500,
                                                                                          32);

    produce_anchor_chains<<<(n_anchors / block_size) + 1, block_size, 0, _cuda_stream>>>(d_anchors.data(),
                                                                                         d_overlaps_source.data(),
                                                                                         d_anchor_scores.data(),
                                                                                         d_overlaps_select_mask.data(),
                                                                                         d_anchor_predecessors.data(),
                                                                                         n_anchors,
                                                                                         40);

    mask_overlaps<<<(n_anchors / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_source.data(),
                                                                                 n_anchors,
                                                                                 d_overlaps_select_mask.data(),
                                                                                 min_overlap_len,
                                                                                 min_residues,
                                                                                 min_bases_per_residue,
                                                                                 all_to_all,
                                                                                 false,
                                                                                 0.8,
                                                                                 0);

    device_buffer<int32_t> d_n_filtered_overlaps(1, _allocator, _cuda_stream);
    drop_overlaps_by_mask(d_overlaps_source,
                          d_overlaps_select_mask,
                          n_anchors,
                          d_overlaps_dest,
                          d_n_filtered_overlaps,
                          _allocator,
                          _cuda_stream);
    int32_t n_filtered_overlaps = cudautils::get_value_from_device(d_n_filtered_overlaps.data(), _cuda_stream);
    d_overlaps_select_mask.clear_and_resize(n_filtered_overlaps);
    init_overlap_mask<<<(n_filtered_overlaps / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_select_mask.data(), n_filtered_overlaps, true);
    mask_overlaps<<<(n_anchors / block_size) + 1, block_size, 0, _cuda_stream>>>(d_overlaps_dest.data(),
                                                                                 n_filtered_overlaps,
                                                                                 d_overlaps_select_mask.data(),
                                                                                 0,
                                                                                 0,
                                                                                 1000000,
                                                                                 all_to_all,
                                                                                 false,
                                                                                 0.8,
                                                                                 16);
    drop_overlaps_by_mask(d_overlaps_dest,
                          d_overlaps_select_mask,
                          n_filtered_overlaps,
                          d_overlaps_source,
                          d_n_filtered_overlaps,
                          _allocator,
                          _cuda_stream);
    n_filtered_overlaps = cudautils::get_value_from_device(d_n_filtered_overlaps.data(), _cuda_stream);

    // device_buffer<double> d_overlap_scores_dest(n_filtered_overlaps, _allocator, _cuda_stream);
    // drop_scores_by_mask(d_anchor_scores,
    //                     d_overlaps_select_mask,
    //                     n_filtered_overlaps,
    //                     d_overlap_scores_dest,
    //                     d_n_filtered_overlaps,
    //                     _allocator,
    //                     _cuda_stream);
    std::cerr << "Writing " << n_filtered_overlaps << " overlaps." << std::endl;
    fused_overlaps.resize(n_filtered_overlaps);
    cudautils::device_copy_n(d_overlaps_source.data(), n_filtered_overlaps, fused_overlaps.data(), _cuda_stream);

    // This is not completely necessary, but if removed one has to make sure that the next step
    // uses the same stream or that sync is done in caller
    GW_CU_CHECK_ERR(cudaStreamSynchronize(_cuda_stream));
}

OverlapperMinimap::OverlapperMinimap(DefaultDeviceAllocator allocator,
                                     const cudaStream_t cuda_stream)
    : _allocator(allocator)
    , _cuda_stream(cuda_stream)
{
}

} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks
