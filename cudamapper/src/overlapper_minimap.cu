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
#include "chainer_utils.cuh"

#ifndef NDEBUG // only needed to check if input is sorted in assert
#include <algorithm>
#include <thrust/host_vector.h>
#endif

// #define DEBUG_CHAINS
//#define CHAINDEBUG

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

#define INT32_INFINITY 100000000
#define NEGATIVE_INT32_INFINITY -1 * INT32_INFINITY
#define PREDECESSOR_SEARCH_ITERATIONS 64
#define PARALLEL_UNITS (BLOCK_COUNT)
#define TILE_SIZE 1024
#define TILING_WINDOW_END (TILE_SIZE + PREDECESSOR_SEARCH_ITERATIONS + 1)
#define THREADS_PER_BLOCK PREDECESSOR_SEARCH_ITERATIONS
#define BLOCK_COUNT 1024
#define BLOCK_SIZE 64

#define MAX_CHAINS_PER_TILE 5

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

__device__ __forceinline__ double
percent_reciprocal_overlap(const Overlap& a, const Overlap& b)
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

// __device__ bool operator==(const Anchor& lhs,
//                            const Anchor& rhs)
// {
//     auto score_threshold = 1;

//     // Very simple scoring function to quantify quality of overlaps.
//     auto score = 1;

//     if (abs(int(rhs.query_position_in_read_) - int(lhs.query_position_in_read_)) <= 50 and
//         abs(int(rhs.target_position_in_read_) - int(lhs.target_position_in_read_)) <= 50)
//         score = 2;
//     if (lhs.query_position_in_read_ == rhs.query_position_in_read_)
//         score = 0;
//     return ((lhs.query_read_id_ == rhs.query_read_id_) &&
//             (lhs.target_read_id_ == rhs.target_read_id_) &&
//             score > score_threshold);
// }

__global__ void mask_overlaps(Overlap* overlaps,
                              bool* select_mask,
                              const int32_t min_overlap_length,
                              const int32_t min_residues,
                              const int32_t max_bases_per_residue,
                              const bool all_to_all,
                              const bool filter_self_mappings,
                              const double max_percent_reciprocal,
                              const int32_t max_reciprocal_iterations,
                              const int32_t n_overlaps)
{
    int32_t d_tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (int i = d_tid; i < n_overlaps; i += stride)
    {
        bool local_select_val                    = select_mask[i];
        Overlap local_current_overlap            = overlaps[i];
        position_in_read_t overlap_query_length  = local_current_overlap.query_end_position_in_read_ - local_current_overlap.query_start_position_in_read_;
        position_in_read_t overlap_target_length = local_current_overlap.target_end_position_in_read_ - local_current_overlap.target_start_position_in_read_;
        //const bool mask_self_self                = overlaps[d_tid].query_read_id_ == overlaps[d_tid].target_read_id_ && all_to_all && filter_self_mappings;
        auto query_bases_per_residue  = static_cast<double>(overlap_query_length) / static_cast<double>(local_current_overlap.num_residues_);
        auto target_bases_per_residue = static_cast<double>(overlap_target_length) / static_cast<double>(local_current_overlap.num_residues_);
        local_select_val              = local_select_val && (overlap_query_length >= min_overlap_length) && (overlap_target_length >= min_overlap_length);
        local_select_val              = local_select_val && (local_current_overlap.num_residues_ >= min_residues);

        local_select_val = local_select_val && (static_cast<int32_t>(query_bases_per_residue) < max_bases_per_residue) && (static_cast<int32_t>(target_bases_per_residue) < max_bases_per_residue);
        // Look at the overlaps and all the overlaps adjacent to me, up to some maximum. Between neighbor i and myself, if
        // some criteria is met, defined by percent_reciprocal_overlap() and contained_overlap(), I get filtered
        // TODO VI: since we do these pair-wise, I think there is some overlap in work that adjacent threads do
        for (int j = d_tid + 1; j < d_tid + max_reciprocal_iterations && j < n_overlaps; ++j)
        {
            Overlap left_neighbor_overlap = overlaps[j];
            local_select_val              = local_select_val && percent_reciprocal_overlap(local_current_overlap, left_neighbor_overlap) <= max_percent_reciprocal;
        }
        select_mask[i] = local_select_val;
    }
}

__device__ __forceinline__ void init_overlap(Overlap& overlap)
{
    overlap.query_read_id_                 = 0;
    overlap.target_read_id_                = 0;
    overlap.query_start_position_in_read_  = UINT32_MAX;
    overlap.query_end_position_in_read_    = 0;
    overlap.target_start_position_in_read_ = UINT32_MAX;
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
        scores[d_tid] = val;
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
                                                  bool* mask, // why is this here??
                                                  size_t n_overlaps)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_overlaps)
    {
        scores[d_tid]       = 0;
        predecessors[d_tid] = -1;
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
                                                            const int32_t word_size,
                                                            const int32_t max_dist,
                                                            const int32_t max_bandwidth)
{
    if (a.query_read_id_ != b.query_read_id_ || a.target_read_id_ != b.target_read_id_)
        return NEGATIVE_INT32_INFINITY;
    if (a.query_position_in_read_ == b.query_position_in_read_ && a.target_position_in_read_ == b.target_position_in_read_)
        return NEGATIVE_INT32_INFINITY;

    int32_t b_query_pos  = b.query_position_in_read_ + word_size;
    int32_t b_target_pos = static_cast<int32_t>(b.target_position_in_read_);
    if (b_target_pos < a.target_position_in_read_)
    {
        b_target_pos -= word_size;
    }
    else
    {
        b_target_pos += word_size;
    }

    //int32_t x_dist = abs(b_target_pos - int(a.target_position_in_read_)); // we might have some type issues here?
    int32_t x_dist = abs(b_target_pos - int(a.target_position_in_read_)); // we might have some type issues here?

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
    // This is sensitive to typecasts...
    if (dist_diff > 0)
        score -= (dist_diff * (0.01 * word_size)) + (log_dist_diff >> 1);
    //printf("%d %d %d %d | %d \n", x_dist, y_dist, min_dist, min_size, score);
    return score;
}

__global__ void chain_anchors_in_tile(const Anchor* anchors,
                                      double* scores,
                                      int32_t* predecessors,
                                      bool* anchor_select_mask,
                                      const int32_t* query_starts,
                                      const int32_t* query_ends,
                                      const int32_t batch_id,
                                      const int32_t num_anchors,
                                      const int32_t num_queries,
                                      const int32_t word_size,
                                      const int32_t max_distance,
                                      const int32_t max_bandwidth)
{
    int32_t block_id           = blockIdx.x;  // Each block processes one read-tile of data.
    int32_t thread_id_in_block = threadIdx.x; // Equivalent to "j." Represents the end of a sliding window.

    int32_t batch_block_id = batch_id * BLOCK_COUNT + block_id;
    // printf("%d %d %d\n", batch_block_id, block_id, batch_size);
    if (batch_block_id < num_queries)
    {
        // printf("%d %d %d\n", batch_block_id, block_id, batch_size);

        int32_t tile_start = query_starts[batch_block_id]; /// tile_starts is length num_tiles
        // All threads in the block will get the same tile_start number

        // write is the leftmost index? global_read_index is offset by my thread_id
        // initialize as the 0th anchor in the tile and the "me" anchor
        int32_t global_write_index = tile_start;
        int32_t global_read_index  = tile_start + thread_id_in_block;
        // printf("%d %d %d %d %d\n", block_id, thread_id_in_block, global_write_index, global_read_index, tile_start);

        // printf("%d %d %d %d\n", block_id, global_write_index, global_read_index, tile_start);
        if (global_write_index < num_anchors)
        {
            __shared__ Anchor block_anchor_cache[PREDECESSOR_SEARCH_ITERATIONS];
            __shared__ bool block_max_select_mask[PREDECESSOR_SEARCH_ITERATIONS];
            __shared__ int32_t block_score_cache[PREDECESSOR_SEARCH_ITERATIONS];
            __shared__ int32_t block_predecessor_cache[PREDECESSOR_SEARCH_ITERATIONS];

            // Initialize the local caches
            // load the current anchor I'm on and it's associated data
            block_anchor_cache[thread_id_in_block]      = anchors[global_read_index];
            block_score_cache[thread_id_in_block]       = static_cast<int32_t>(scores[global_read_index]);
            block_predecessor_cache[thread_id_in_block] = predecessors[global_read_index];
            block_max_select_mask[thread_id_in_block]   = false;

            // iterate through the tile
            // VI: We do iterate i throughout
            // Do we go over to the next tile here...? Is that ok?
            // Answer to the above is it should be fine (maybe)
            int32_t i       = PREDECESSOR_SEARCH_ITERATIONS;
            int32_t counter = 0;
            int32_t current_score;
            int32_t current_pred;
            bool current_mask;
            for (; tile_start + counter < query_ends[batch_block_id]; counter++, i++)
            {
                __syncthreads();
                // on the first iteration, every thread looks at the 0th anchor
                const Anchor possible_successor_anchor = block_anchor_cache[i % PREDECESSOR_SEARCH_ITERATIONS];
                current_score                          = block_score_cache[i % PREDECESSOR_SEARCH_ITERATIONS];
                current_pred                           = block_predecessor_cache[i % PREDECESSOR_SEARCH_ITERATIONS];
                current_mask                           = block_max_select_mask[i % PREDECESSOR_SEARCH_ITERATIONS];
                if (current_score <= word_size)
                {
                    current_score = word_size;
                    current_pred  = -1;
                    current_mask  = false;
                }
                __syncthreads();

                // I think this is the thread at the LHS of the window
                if ((thread_id_in_block == (i % PREDECESSOR_SEARCH_ITERATIONS)) && (global_read_index + i < num_anchors))
                {
                    // Implies that the thread is at the left_side (head, front of FIFO queue) of a window
                    // Read in the anchor, score, and predecessor of the next anchor in memory.
                    // I think we load if we're at the LHS so we want to get the (threadIdx + 64)th anchor
                    block_anchor_cache[thread_id_in_block]      = anchors[global_write_index + i];
                    block_score_cache[thread_id_in_block]       = 0;
                    block_predecessor_cache[thread_id_in_block] = -1;
                    block_max_select_mask[thread_id_in_block]   = false;
                }
                else if (thread_id_in_block == (i % PREDECESSOR_SEARCH_ITERATIONS))
                {
                    block_anchor_cache[thread_id_in_block]      = chainerutils::empty_anchor();
                    block_score_cache[thread_id_in_block]       = 0;
                    block_predecessor_cache[thread_id_in_block] = -1;
                    block_max_select_mask[thread_id_in_block]   = false;
                }

                __syncthreads();
                // Calculate score order matters here
                int32_t marginal_score = log_linear_anchor_weight(possible_successor_anchor, block_anchor_cache[thread_id_in_block], word_size, max_distance, max_bandwidth);

                __syncthreads();

                if ((current_score + marginal_score >= block_score_cache[thread_id_in_block]) && (global_read_index + i < num_anchors))
                {
                    //current_score                               = current_score + marginal_score;
                    //current_pred                                = tile_start + counter - 1;
                    block_score_cache[thread_id_in_block] = current_score + marginal_score;
                    // TODO VI: I'm not entirely sure about this part
                    block_predecessor_cache[thread_id_in_block] = tile_start + counter; // this expands to tile_starts[batch_block_id] + counter, where counter is 0 -> 1024
                    if (current_score + marginal_score > word_size)
                    {
                        block_max_select_mask[thread_id_in_block]                = true;
                        block_max_select_mask[i % PREDECESSOR_SEARCH_ITERATIONS] = false;
                    }

                    //block_max_select_mask[i % PREDECESSOR_SEARCH_ITERATIONS] = false;
                }
                __syncthreads();

                // I'm leftmost thread in teh sliding window, so I write my result out from cache to global array
                if (thread_id_in_block == counter % PREDECESSOR_SEARCH_ITERATIONS && (global_write_index + counter) < num_anchors)
                {
                    // Position thread_id_in_block is at the left-side (tail) of the window.
                    // It has therefore completed n = PREDECESSOR_SEARCH_ITERATIONS iterations.
                    // It's final score is known.
                    // Write its score and predecessor to the global_write_index + counter.
                    scores[global_write_index + counter] = static_cast<double>(current_score);
                    //predecessors[global_write_index + counter]       = block_predecessor_cache[thread_id_in_block];
                    predecessors[global_write_index + counter] = current_pred;
                    //anchor_select_mask[global_write_index + counter] = block_max_select_mask[thread_id_in_block];
                    anchor_select_mask[global_write_index + counter] = current_mask;
                }
            }
            __syncthreads();
            // TODO not sure if this is correct
            if (global_write_index + counter + thread_id_in_block < num_anchors)
            {
                scores[global_write_index + counter + thread_id_in_block]             = block_score_cache[thread_id_in_block % i];
                predecessors[global_write_index + counter + thread_id_in_block]       = block_predecessor_cache[thread_id_in_block % i];
                anchor_select_mask[global_write_index + counter + thread_id_in_block] = block_max_select_mask[thread_id_in_block];
                // scores[global_write_index + counter + thread_id_in_block]             = current_score;
                // predecessors[global_write_index + counter + thread_id_in_block]       = current_pred;
                // anchor_select_mask[global_write_index + counter + thread_id_in_block] = current_mask;
            }
        }
    }
}

///
/// \brief Chains one tile of anchors within a read-tile
/// TODO VI: I don't know if this is true
/// This tile may contain anchors from a single read or multiple
/// reads.
/// Each block starts within one reads and proceeds tile_size iterations
/// forward, sliding a window size PREDECESSOR_SEARCH_ITERATIONS along
/// a tile of TILE_SIZE anchors.
///
/// \param anchors
/// \param scores
/// \param predecessors
/// \param anchor_select_mask
/// \param query_starts
/// \param query_lengths
/// \param query_ends
/// \param tiles_per_query_id
/// \param num_anchors
/// \param batch_id
/// \param batch_size
/// \param word_size
/// \param max_distance
/// \param max_bandwidth
/// \return __global__

// Each block operates on a single tile
// TODO VI: Do we need to pad the anchors array (with null tiles) so it's tile aligned? We may be missing out on the anchors in the last tile
__global__ void chain_anchors_in_block(const Anchor* anchors,
                                       double* scores,
                                       int32_t* predecessors,
                                       bool* anchor_select_mask,
                                       const int32_t* tile_starts,
                                       const int32_t num_anchors,
                                       const int32_t num_query_tiles,
                                       const int32_t batch_id,   // which batch number we are on
                                       const int32_t batch_size, // fixed to TILE_SIZE...?
                                       const int32_t word_size,
                                       const int32_t max_distance,
                                       const int32_t max_bandwidth)
{
    int32_t block_id           = blockIdx.x;  // Each block processes one read-tile of data.
    int32_t thread_id_in_block = threadIdx.x; // Equivalent to "j." Represents the end of a sliding window.

    // figure out which tile I am currently working on
    int32_t batch_block_id = batch_id * batch_size + block_id;
    if (batch_block_id < num_query_tiles)
    {

        // All threads in the block will get the same tile_start number
        int32_t tile_start = tile_starts[batch_block_id];

        // write is the leftmost index? global_read_index is offset by my thread_id
        // initialize as the 0th anchor in the tile and the "me" anchor
        int32_t global_write_index = tile_start;
        int32_t global_read_index  = tile_start + thread_id_in_block;
        if (global_write_index < num_anchors)
        {
            __shared__ Anchor block_anchor_cache[PREDECESSOR_SEARCH_ITERATIONS];
            __shared__ bool block_max_select_mask[PREDECESSOR_SEARCH_ITERATIONS];
            __shared__ int32_t block_score_cache[PREDECESSOR_SEARCH_ITERATIONS];
            __shared__ int32_t block_predecessor_cache[PREDECESSOR_SEARCH_ITERATIONS];

            // Initialize the local caches
            // load the current anchor I'm on and it's associated data
            block_anchor_cache[thread_id_in_block]      = anchors[global_read_index];
            block_score_cache[thread_id_in_block]       = static_cast<int32_t>(scores[global_read_index]);
            block_predecessor_cache[thread_id_in_block] = predecessors[global_read_index];
            block_max_select_mask[thread_id_in_block]   = false;

            // iterate through the tile
            // VI: We do iterate i throughout
            // Do we go over to the next tile here...? Is that ok?
            // Answer to the above is it should be fine (maybe)
            int32_t i       = PREDECESSOR_SEARCH_ITERATIONS;
            int32_t counter = 0;
            int32_t current_score;
            int32_t current_pred;
            bool current_mask;
            for (; counter < batch_size; counter++, i++)
            {
                __syncthreads();
                // on the first iteration, every thread looks at the 0th anchor
                const Anchor possible_successor_anchor = block_anchor_cache[i % PREDECESSOR_SEARCH_ITERATIONS];
                current_score                          = block_score_cache[i % PREDECESSOR_SEARCH_ITERATIONS];
                current_pred                           = block_predecessor_cache[i % PREDECESSOR_SEARCH_ITERATIONS];
                current_mask                           = block_max_select_mask[i % PREDECESSOR_SEARCH_ITERATIONS];
                if (current_score <= word_size)
                {
                    current_score = word_size;
                    current_pred  = -1;
                    current_mask  = false;
                }
                __syncthreads();

                // I think this is the thread at the LHS of the window
                if ((thread_id_in_block == (i % PREDECESSOR_SEARCH_ITERATIONS)) && (global_read_index + i < num_anchors))
                {
                    // Implies that the thread is at the left_side (head, front of FIFO queue) of a window
                    // Read in the anchor, score, and predecessor of the next anchor in memory.
                    // I think we load if we're at the LHS so we want to get the (threadIdx + 64)th anchor
                    block_anchor_cache[thread_id_in_block]      = anchors[global_write_index + i];
                    block_score_cache[thread_id_in_block]       = 0;
                    block_predecessor_cache[thread_id_in_block] = -1;
                    block_max_select_mask[thread_id_in_block]   = false;
                }
                else if (thread_id_in_block == (i % PREDECESSOR_SEARCH_ITERATIONS))
                {
                    block_anchor_cache[thread_id_in_block]      = chainerutils::empty_anchor();
                    block_score_cache[thread_id_in_block]       = 0;
                    block_predecessor_cache[thread_id_in_block] = -1;
                    block_max_select_mask[thread_id_in_block]   = false;
                }

                __syncthreads();
                // Calculate score order matters here
                int32_t marginal_score = log_linear_anchor_weight(possible_successor_anchor, block_anchor_cache[thread_id_in_block], word_size, max_distance, max_bandwidth);

                __syncthreads();

                if ((current_score + marginal_score >= block_score_cache[thread_id_in_block]) && (global_read_index + i < num_anchors))
                {
                    //current_score                               = current_score + marginal_score;
                    //current_pred                                = tile_start + counter - 1;
                    block_score_cache[thread_id_in_block] = current_score + marginal_score;
                    // TODO VI: I'm not entirely sure about this part
                    block_predecessor_cache[thread_id_in_block] = tile_start + counter; // this expands to tile_starts[batch_block_id] + counter, where counter is 0 -> 1024
                    if (current_score + marginal_score > word_size)
                    {
                        block_max_select_mask[thread_id_in_block]                = true;
                        block_max_select_mask[i % PREDECESSOR_SEARCH_ITERATIONS] = false;
                    }

                    //block_max_select_mask[i % PREDECESSOR_SEARCH_ITERATIONS] = false;
                }
                __syncthreads();

                // I'm leftmost thread in teh sliding window, so I write my result out from cache to global array
                if (thread_id_in_block == counter % PREDECESSOR_SEARCH_ITERATIONS && (global_write_index + counter) < num_anchors)
                {
                    // Position thread_id_in_block is at the left-side (tail) of the window.
                    // It has therefore completed n = PREDECESSOR_SEARCH_ITERATIONS iterations.
                    // It's final score is known.
                    // Write its score and predecessor to the global_write_index + counter.
                    scores[global_write_index + counter] = static_cast<double>(current_score);
                    //predecessors[global_write_index + counter]       = block_predecessor_cache[thread_id_in_block];
                    predecessors[global_write_index + counter] = current_pred;
                    //anchor_select_mask[global_write_index + counter] = block_max_select_mask[thread_id_in_block];
                    anchor_select_mask[global_write_index + counter] = current_mask;
                }
            }
            __syncthreads();
            // TODO not sure if this is correct
            if (global_write_index + counter + thread_id_in_block < num_anchors)
            {
                scores[global_write_index + counter + thread_id_in_block]             = current_score;
                predecessors[global_write_index + counter + thread_id_in_block]       = current_pred;
                anchor_select_mask[global_write_index + counter + thread_id_in_block] = current_mask;
            }
            // printf("%d %d %d %d | %f | %d", );
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

__global__ void produce_anchor_chains(const Anchor* anchors,
                                      Overlap* overlaps,
                                      double* scores,
                                      bool* max_select_mask,
                                      int32_t* predecessors,
                                      const int32_t n_anchors,
                                      const int32_t min_score)
{
    const std::size_t d_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_tid < n_anchors)
    {
        // printf("Anchor ID: %d, %c, %d, %d\n", static_cast<int>(d_tid), (max_select_mask[d_tid] ? 'M' : '.'), static_cast<int>(scores[d_tid]), predecessors[d_tid]);
        if (max_select_mask[d_tid] && scores[d_tid] >= min_score)
        {
            int32_t global_overlap_index = d_tid;
            int32_t index                = global_overlap_index;
            Overlap final_overlap        = overlaps[global_overlap_index];
            Anchor first_anchor          = anchors[global_overlap_index];
            double final_score           = scores[global_overlap_index];
            init_overlap(final_overlap);
            add_anchor_to_overlap(anchors[global_overlap_index], final_overlap);
            while (index != -1 && index != predecessors[index])
            {
                int32_t pred = predecessors[index];
                if (pred != -1)
                {
                    add_anchor_to_overlap(anchors[pred], final_overlap);
                    // printf("| %d %d %d %d | -> | %d %d %d %d |\n", anchors[index].query_read_id_, anchors[index].query_position_in_read_, anchors[index].target_read_id_, anchors[index].target_position_in_read_,
                    //        anchors[pred].query_read_id_, anchors[pred].query_position_in_read_, anchors[pred].target_read_id_, anchors[pred].target_position_in_read_);
                    max_select_mask[pred] = false;
                }
                index = predecessors[index];
            }
            overlaps[global_overlap_index] = final_overlap;
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
    int32_t n_filtered = cudautils::get_value_from_device(d_filtered_count.data(), _cuda_stream);
    d_mask.clear_and_resize(n_filtered);
    chainerutils::initialize_mask<<<BLOCK_COUNT, BLOCK_SIZE, 0, _cuda_stream>>>(d_mask.data(), n_filtered, true);
}

void OverlapperMinimap::get_overlaps(std::vector<Overlap>& fused_overlaps,
                                     const device_buffer<Anchor>& d_anchors,
                                     bool all_to_all,
                                     int64_t min_residues,
                                     int64_t min_overlap_len,
                                     int64_t min_bases_per_residue,
                                     float min_overlap_fraction)
{
    const int32_t n_anchors = d_anchors.size();

    device_buffer<bool> d_overlaps_select_mask(n_anchors, _allocator, _cuda_stream);
    device_buffer<Overlap> d_overlaps_source(n_anchors, _allocator, _cuda_stream);
    device_buffer<Overlap> d_overlaps_dest(n_anchors, _allocator, _cuda_stream);

    device_buffer<int32_t> d_anchor_predecessors(n_anchors, _allocator, _cuda_stream);
    device_buffer<double> d_anchor_scores(n_anchors, _allocator, _cuda_stream);

    chainerutils::initialize_mask<<<BLOCK_COUNT, BLOCK_SIZE, 0, _cuda_stream>>>(d_overlaps_select_mask.data(), n_anchors, true);
    chainerutils::initialize_array<<<BLOCK_COUNT, BLOCK_SIZE, 0, _cuda_stream>>>(d_anchor_scores.data(), n_anchors, 0.0);
    chainerutils::initialize_array<<<BLOCK_COUNT, BLOCK_SIZE, 0, _cuda_stream>>>(d_anchor_predecessors.data(), n_anchors, -1);

    // Allocate half the number of anchors for the number of QT pairs under the naive assumption that
    // the average number of anchors per read is greater than 2 (i.e., the total number of Anchors >> 2 times the total number of Q-T pairs).
    device_buffer<int32_t> query_id_starts(n_anchors, _allocator, _cuda_stream);
    device_buffer<int32_t> query_id_lengths(n_anchors, _allocator, _cuda_stream);
    device_buffer<int32_t> query_id_ends(n_anchors, _allocator, _cuda_stream);

    int32_t n_queries;

    // generates the scheduler blocks
    chainerutils::encode_query_locations_from_anchors(d_anchors.data(),
                                                      n_anchors,
                                                      query_id_starts,
                                                      query_id_lengths,
                                                      query_id_ends,
                                                      n_queries,
                                                      _allocator,
                                                      _cuda_stream); // This is threads per block

    int32_t num_batches = (n_queries / BLOCK_COUNT) + 1;
    std::cerr << "N_anchors = " << n_anchors << std::endl;
    std::cerr << "N_queries = " << n_queries << std::endl;
    for (std::size_t batch = 0; batch < static_cast<size_t>(num_batches); ++batch)
    {
        std::cerr << "Processing batch " << batch << " of " << num_batches << ". Total queries: " << n_queries << "." << std::endl;
        // Launch one 1792 blocks (from paper), with 64 threads
        // each batch is of size BLOCK_COUNT
        chain_anchors_in_tile<<<BLOCK_COUNT, PREDECESSOR_SEARCH_ITERATIONS, 0, _cuda_stream>>>(d_anchors.data(),
                                                                                               d_anchor_scores.data(),
                                                                                               d_anchor_predecessors.data(),
                                                                                               d_overlaps_select_mask.data(),
                                                                                               query_id_starts.data(),
                                                                                               query_id_ends.data(),
                                                                                               batch,
                                                                                               n_anchors,
                                                                                               n_queries,
                                                                                               15,
                                                                                               5000,
                                                                                               500);
    }

// #define DEBUG_CHAINS
#ifdef DEBUG_CHAINS
    std::cout << "Num anchors: " << n_anchors << std::endl;
    std::cout << "Num query tiles: " << n_query_tiles << std::endl;
    std::cout << "Num batches: " << num_batches << std::endl;

    // fetch the computed data from the chaining algorithm
    std::vector<double> chain_scores;
    std::vector<int32_t> predecessors;
    chain_scores.resize(n_anchors);
    predecessors.resize(n_anchors);
    cudautils::device_copy_n(d_anchor_scores.data(), n_anchors, chain_scores.data(), _cuda_stream);
    //predecessors.resize(n_anchors);
    cudautils::device_copy_n(d_anchor_predecessors.data(), n_anchors, predecessors.data(), _cuda_stream);
    for (std::size_t i = 0; i < chain_scores.size(); ++i)
    {
        std::cout << i << "\t" << chain_scores[i] << "\t" << predecessors[i] << std::endl;
    }
#endif

    // the deschedule block. Get outputs from here
    chainerutils::backtrace_anchors_to_overlaps<<<BLOCK_COUNT, BLOCK_SIZE, 0, _cuda_stream>>>(d_anchors.data(),
                                                                                              d_overlaps_source.data(),
                                                                                              d_anchor_scores.data(),
                                                                                              d_overlaps_select_mask.data(),
                                                                                              d_anchor_predecessors.data(),
                                                                                              n_anchors,
                                                                                              40);

    // TODO VI: I think we can get better device occupancy here with some kernel refactoring
    mask_overlaps<<<BLOCK_COUNT, BLOCK_SIZE, 0, _cuda_stream>>>(d_overlaps_source.data(),
                                                                d_overlaps_select_mask.data(),
                                                                min_overlap_len,
                                                                min_residues,
                                                                min_bases_per_residue,
                                                                all_to_all,
                                                                false,
                                                                0.9,
                                                                50,
                                                                n_anchors);
    device_buffer<int32_t> d_n_filtered_overlaps(1, _allocator, _cuda_stream);
    drop_overlaps_by_mask(d_overlaps_source,
                          d_overlaps_select_mask,
                          n_anchors,
                          d_overlaps_dest,
                          d_n_filtered_overlaps,
                          _allocator,
                          _cuda_stream);
    int32_t n_filtered_overlaps = cudautils::get_value_from_device(d_n_filtered_overlaps.data(), _cuda_stream);
    std::cerr << "Number of filtered chains: " << n_filtered_overlaps << std::endl;

    fused_overlaps.resize(n_filtered_overlaps);
    cudautils::device_copy_n(d_overlaps_dest.data(), n_filtered_overlaps, fused_overlaps.data(), _cuda_stream);

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
