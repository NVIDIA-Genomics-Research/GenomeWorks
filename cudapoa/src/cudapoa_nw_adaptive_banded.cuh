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

#pragma once

#include "cudapoa_structs.cuh"

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/limits.cuh>

#include <stdio.h>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/**
 * @brief Device function for getting the address of an element specified by (row, column) in the score matrix 
 *        taking adaptive band-width into consideration.
 *        
 * @param[in] scores              Score matrix
 * @param[in] row                 Row # of the element
 * @param[in] column              Column # of the element
 * @param[in] band_starts         Array of band_starts per row
 * @param[in] head_indices        Array of indexes in score that map to band_start per row
 * @param[out] score_address      Address of the element indicated by the (row, col) tuple
*/

template <typename ScoreT, typename SizeT>
__device__ ScoreT* get_score_ptr_adaptive(ScoreT* scores, SizeT row, SizeT column, SizeT band_start, int64_t* head_indices)
{
    column              = column == -1 ? 0 : column - band_start;
    int64_t score_index = static_cast<int64_t>(column) + head_indices[row];
    return &scores[score_index];
};

/**
 * @brief Device function for setting an element specified by (row, column) in the score matrix taking adaptive band-width into consideration.
 *        
 * @param[in] scores              Score matrix
 * @param[in] row                 Row # of the element
 * @param[in] column              Column # of the element
 * @param[in] value               Value to set
 * @param[in] band_starts         Array of band_starts per row
 * @param[in] head_indices        Array of indexes in score that map to band_start per row
*/
template <typename ScoreT, typename SizeT>
__device__ void set_score_adaptive(ScoreT* scores, SizeT row, SizeT column, ScoreT value, SizeT* band_starts, int64_t* head_indices)
{
    SizeT band_start = band_starts[row];

    SizeT col_offset;
    if (column == -1)
    {
        col_offset = band_start;
    }
    else
    {
        col_offset = column - band_start;
    }

    int64_t score_index = static_cast<int64_t>(col_offset) + head_indices[row];

    scores[score_index] = value;
}

template <typename ScoreT, typename SizeT>
__device__ void initialize_band_adaptive(ScoreT* scores, SizeT row, ScoreT min_score_value, SizeT* band_starts, SizeT* band_widths, int64_t* head_indices, SizeT max_column)
{
    SizeT lane_idx   = threadIdx.x % WARP_SIZE;
    SizeT band_start = band_starts[row];

    SizeT band_end = band_start + band_widths[row];
    band_start     = max(1, band_starts[row]);

    set_score_adaptive(scores, row, band_start, min_score_value, band_starts, head_indices);
    if (lane_idx < CUDAPOA_BANDED_MATRIX_RIGHT_PADDING)
    {
        set_score_adaptive(scores, row, static_cast<SizeT>(lane_idx + band_end), min_score_value, band_starts, head_indices);
    }
};

/**
 * @brief Device function for getting an element specified by (row, column) in the score matrix taking adaptive band-width into consideration.
 *        
 * @param[in] scores              Score matrix
 * @param[in] row                 Row # of the element
 * @param[in] column              Column # of the element
 * @param[in] value               Value to set
 * @param[in] band_starts         Array of band_starts per row
 * @param[in] band_widths         Array of band_widths per row
 * @param[in] head_indices        Array of indexes in score that map to band_start per row
 * @param[in] max_column          Last column # in the score matrix
 * @param[in] min_score_value     Minimum score value
 * @param[out] score              Score at the specified row and column
*/
template <typename ScoreT, typename SizeT>
__device__ ScoreT get_score_adaptive(ScoreT* scores, SizeT row, SizeT column, SizeT* band_starts, SizeT* band_widths, int64_t* head_indices, SizeT max_column, const ScoreT min_score_value)
{
    SizeT band_start = band_starts[row];
    SizeT band_end   = band_start + band_widths[row];

    if ((column > band_end || column < band_start) && column != -1)
    {
        return min_score_value;
    }
    else
    {
        return *get_score_ptr_adaptive(scores, row, column, band_start, head_indices);
    }
}

template <typename ScoreT, typename SizeT>
__device__ ScoreT4<ScoreT> get_scores_adaptive(ScoreT* scores,
                                               SizeT row,
                                               SizeT column,
                                               SizeT* band_starts,
                                               SizeT* band_widths,
                                               int64_t* head_indices,
                                               SizeT max_column,
                                               ScoreT default_value,
                                               ScoreT gap_score,
                                               ScoreT4<ScoreT>& char_profile)
{

    // The load instructions typically load data in 4B or 8B chunks.
    // If data is 16b (2B), then a 4B load chunk is loaded into register
    // and the necessary bits are extracted before returning. This wastes cycles
    // as each read of 16b issues a separate load command.
    // Instead it is better to load a 4B or 8B chunk into a register
    // using a single load inst, and then extracting necessary part of
    // of the data using bit arithmatic. Also reduces register count.

    SizeT band_start = band_starts[row];

    // subtract by CELLS_PER_THREAD to ensure score4_next is not pointing out of the corresponding band bounds
    SizeT band_end = static_cast<SizeT>(band_start + band_widths[row] - CELLS_PER_THREAD);

    if ((column > band_end || column < band_start) && column != -1)
    {
        return ScoreT4<ScoreT>{default_value, default_value, default_value, default_value};
    }
    else
    {
        ScoreT4<ScoreT>* pred_scores = (ScoreT4<ScoreT>*)get_score_ptr_adaptive(scores, row, column, band_start, head_indices);

        // loads 8 consecutive bytes (4 shorts)
        ScoreT4<ScoreT> score4 = pred_scores[0];

        // need to load the next chunk of memory as well
        ScoreT4<ScoreT> score4_next = pred_scores[1];

        ScoreT4<ScoreT> score;

        score.s0 = max(score4.s0 + char_profile.s0,
                       score4.s1 + gap_score);
        score.s1 = max(score4.s1 + char_profile.s1,
                       score4.s2 + gap_score);
        score.s2 = max(score4.s2 + char_profile.s2,
                       score4.s3 + gap_score);
        score.s3 = max(score4.s3 + char_profile.s3,
                       score4_next.s0 + gap_score);

        return score;
    }
}

// The following kernel finds index and value of the maximum score in 32 consequtive cells in a given row
// if two or more scores are equal to the maximum value, the left most index will be output
template <typename ScoreT, typename SizeT>
__device__ void warp_reduce_max(ScoreT& val, SizeT& idx)
{
    for (int16_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        ScoreT tmp_val = __shfl_down_sync(FULL_MASK, val, offset);
        SizeT tmp_idx  = __shfl_down_sync(FULL_MASK, idx, offset);
        if (tmp_val > val)
        {
            val = tmp_val;
            idx = tmp_idx;
        }
    }
}

template <typename ScoreT, typename SizeT>
__device__ void get_predecessors_max_score_index(SizeT& pred_max_score_left,
                                                 SizeT& pred_max_score_right,
                                                 SizeT row,
                                                 SizeT node_id,
                                                 ScoreT* scores,
                                                 SizeT* node_id_to_pos,
                                                 SizeT* max_indices,
                                                 SizeT* band_widths,
                                                 SizeT* band_starts,
                                                 uint16_t* incoming_edge_count,
                                                 SizeT* incoming_edges,
                                                 int64_t* head_indices,
                                                 ScoreT min_score_value)
{
    for (uint16_t p = 0; p < incoming_edge_count[node_id]; p++)
    {
        SizeT pred_idx       = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]];
        SizeT max_score_idx  = max_indices[pred_idx];
        ScoreT max_score_val = min_score_value;
        int64_t head_index   = head_indices[pred_idx];

        // max score index for this (predecessor) node_id is not computed yet
        if (max_score_idx == -1)
        {
            SizeT lane_idx   = threadIdx.x % WARP_SIZE;
            SizeT band_width = band_widths[pred_idx];
            SizeT band_start = band_starts[pred_idx];

            for (SizeT index = lane_idx; index < band_width; index += WARP_SIZE)
            {
                ScoreT score_val = scores[static_cast<int64_t>(index) + head_index];
                SizeT score_idx  = index + band_start;
                warp_reduce_max(score_val, score_idx);
                score_val = __shfl_sync(FULL_MASK, score_val, 0);
                if (score_val > max_score_val)
                {
                    max_score_val = score_val;
                    max_score_idx = __shfl_sync(FULL_MASK, score_idx, 0);
                }
            }
            max_indices[pred_idx] = max_score_idx;
        }

        pred_max_score_left  = max_score_idx < pred_max_score_left ? max_score_idx : pred_max_score_left;
        pred_max_score_right = max_score_idx > pred_max_score_right ? max_score_idx : pred_max_score_right;
    }
}

template <typename ScoreT, typename SizeT>
__device__
    ScoreT
    set_and_get_first_column_score(SizeT* node_id_to_pos,
                                   SizeT node_id,
                                   SizeT row,
                                   ScoreT* scores,
                                   uint16_t* incoming_edge_count,
                                   SizeT* incoming_edges,
                                   SizeT* band_starts,
                                   SizeT* band_widths,
                                   int64_t* head_indices,
                                   SizeT max_column,
                                   ScoreT score_type_min_limit,
                                   ScoreT min_score_value,
                                   ScoreT gap_score)
{
    ScoreT first_column_score;
    uint16_t pred_count = incoming_edge_count[node_id];
    if (pred_count == 0)
    {
        first_column_score = gap_score;
        set_score_adaptive(scores, row, SizeT{-1}, first_column_score, band_starts, head_indices);
    }
    else
    {
        ScoreT penalty = score_type_min_limit;
        for (uint16_t p = 0; p < pred_count; p++)
        {
            SizeT pred_node_id        = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
            SizeT pred_node_graph_pos = node_id_to_pos[pred_node_id] + 1;
            penalty                   = max(penalty, get_score_adaptive(scores, pred_node_graph_pos, SizeT{-1}, band_starts, band_widths, head_indices, max_column, min_score_value));
        }
        first_column_score = penalty + gap_score;
        set_score_adaptive(scores, row, SizeT{-1}, first_column_score, band_starts, head_indices);
    }

    return first_column_score;
}

template <typename SizeT, typename ScoreT>
__device__ SizeT set_band_parameters(ScoreT* scores,
                                    int64_t scores_size,
                                    SizeT* band_starts,
                                    SizeT* band_widths,
                                    int64_t* head_indices,
                                    SizeT* max_indices,
                                    uint16_t* incoming_edge_count,
                                    SizeT* incoming_edges,
                                    SizeT* node_distances,
                                    SizeT* node_id_to_pos,
                                    int64_t& head_index,
                                    SizeT node_id,
                                    SizeT row,
                                    SizeT max_column,
                                    SizeT graph_length,
                                    float gradient,
                                    ScoreT min_score_value)
{
    SizeT err = 0;
    SizeT pred_max_score_left  = max_column;
    SizeT pred_max_score_right = 0;

    get_predecessors_max_score_index(pred_max_score_left, pred_max_score_right, row, node_id, scores, node_id_to_pos, max_indices, band_widths,
                                     band_starts, incoming_edge_count, incoming_edges, head_indices, min_score_value);

    SizeT node_distance_i = node_distances[node_id];

    SizeT b_start = min(node_distance_i, pred_max_score_left);
    b_start       = b_start < 0 ? 0 : b_start;
    SizeT b_end   = max(node_distance_i, pred_max_score_right);
    b_end         = b_end > max_column ? max_column : b_end;

    //bandwidth should be multiple of CUDAPOA_MIN_BAND_WIDTH
    SizeT bw             = (b_end - b_start) > 0 ? b_end - b_start : 1;
    SizeT band_width     = cudautils::align<SizeT, CUDAPOA_MIN_BAND_WIDTH>(bw);
    SizeT extended_width = band_width - bw;

    SizeT start_pos = b_start - extended_width / 2;
    start_pos       = max(start_pos, 0);

    SizeT end_pos = start_pos + band_width;

    if (end_pos > max_column)
    {
        start_pos  = max_column - band_width + CELLS_PER_THREAD;
        band_width = cudautils::align<SizeT, CUDAPOA_MIN_BAND_WIDTH>(bw);
    }

    // there is no guarantee that end_pos does not fall on the left side of the main diagonal, therefore in some cases
    // for the last node, adaptive band may not be covering right bottom corner of the score matrix, i.e.
    // for the last node, end_pos < max_column. For global alignment, this should be avoided, therefore we add the following modification
    SizeT diagonal_index = SizeT(row * gradient) + 1;
    if (end_pos < diagonal_index)
    {
        bw             = diagonal_index - b_start;
        band_width     = cudautils::align<SizeT, CUDAPOA_MIN_BAND_WIDTH>(bw);
        extended_width = band_width - bw;
        start_pos      = b_start - extended_width / 2;
    }

    start_pos = max(start_pos, 0);
    start_pos = start_pos - (start_pos % CELLS_PER_THREAD);

    //if (threadIdx.x == 0)
    //{
    //    printf("-> %3d %3d (pl %3d, pr %3d) , (bl %3d, br %3d) , bw %3d,   dist %3d\n", row, node_id,
    //    pred_max_score_left, pred_max_score_right, start_pos, end_pos, band_width, node_distance_i);
    //}

    band_starts[row]  = start_pos;
    band_widths[row]  = band_width;
    head_indices[row] = head_index;
    

    // update head_index for the nex row
    head_index += static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
    if (head_index > scores_size)
    {
        // If current end of band is greater than allocated scores' size, return error
        err = -2;
    }
    return err;
}

template <typename SeqT,
          typename ScoreT,
          typename SizeT>
__device__
    SizeT
    runNeedlemanWunschAdaptiveBanded(SeqT* nodes,
                                     SizeT* graph,
                                     SizeT* node_id_to_pos,
                                     SizeT graph_count,
                                     uint16_t* incoming_edge_count,
                                     SizeT* incoming_edges,
                                     uint16_t* outgoing_edge_count,
                                     SeqT* read,
                                     SizeT read_length,
                                     ScoreT* scores,
                                     int64_t scores_size,
                                     SizeT* alignment_graph,
                                     SizeT* alignment_read,
                                     SizeT* node_distances,
                                     SizeT* band_starts,
                                     SizeT* band_widths,
                                     int64_t* head_indices,
                                     SizeT* max_indices,
                                     SizeT static_band_width,
                                     ScoreT gap_score,
                                     ScoreT mismatch_score,
                                     ScoreT match_score)
{

    GW_CONSTEXPR ScoreT score_type_min_limit = numeric_limits<ScoreT>::min();
    // in adaptive bands, there are cases where multiple rows happen to have a band with start index
    // smaller than band-start index of a row above. If min_value is too close to score_type_min_limit,
    // this can cause overflow, therefore min_score_value is selected far enough
    const ScoreT min_score_value = score_type_min_limit / 2;

    int16_t lane_idx = threadIdx.x % WARP_SIZE;
    int64_t score_index;

    //Calculate gradient for the scores matrix
    float gradient = float(read_length + 1) / float(graph_count + 1);

    SizeT max_column = read_length + 1;

    SizeT max_matrix_sequence_dimension = static_band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING;

    // set band parameters for node_id 0 (row 0)
    band_starts[0]  = 0;
    band_widths[0]  = static_band_width;
    head_indices[0] = 0;
    // will be used in set_band_parameters() to update head_indices array
    int64_t head_index_of_next_row = max_matrix_sequence_dimension;

    // Initialise the horizontal boundary of the score matrix, initialising of the vertical boundary is done within the main for loop
    for (SizeT j = lane_idx; j < max_matrix_sequence_dimension; j += WARP_SIZE)
    {
        set_score_adaptive(scores, SizeT{0}, j, static_cast<ScoreT>(j * gap_score), band_starts, head_indices);
    }

    // reset max score indices per row
    for (SizeT i = lane_idx; i < graph_count; i += WARP_SIZE)
    {
        max_indices[i] = -1;
    }

#ifdef NW_VERBOSE_PRINT
    if (lane_idx == 0)
    {
        printf("graph %d, read %d\n", graph_count, read_length);
    }
#endif

    __syncwarp();

    SeqT4<SeqT>* d_read4 = (SeqT4<SeqT>*)read;

    // compute vertical and diagonal values in parallel.
    for (SizeT graph_pos = 0; graph_pos < graph_count; graph_pos++)
    {

        SizeT node_id    = graph[graph_pos];
        SizeT score_gIdx = graph_pos + 1;
        
        SizeT err = set_band_parameters(scores, scores_size, band_starts, band_widths, head_indices, max_indices, incoming_edge_count, incoming_edges, node_distances,
                                        node_id_to_pos, head_index_of_next_row, node_id, score_gIdx, max_column, graph_count, gradient, min_score_value);
        if(err)
        {
            return err;
        }

        SizeT band_start = band_starts[score_gIdx];

        initialize_band_adaptive(scores, score_gIdx, min_score_value, band_starts, band_widths, head_indices, max_column);

        ScoreT first_element_prev_score = 0;
        if (lane_idx == 0)
        {
            first_element_prev_score = set_and_get_first_column_score(node_id_to_pos, node_id, score_gIdx, scores, incoming_edge_count, incoming_edges, band_starts,
                                                                      band_widths, head_indices, max_column, score_type_min_limit, min_score_value, gap_score);
        }

        uint16_t pred_count = incoming_edge_count[node_id];

        SizeT pred_idx = (pred_count == 0 ? 0 : node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1);

        SeqT graph_base = nodes[node_id];

        for (SizeT read_pos = lane_idx * CELLS_PER_THREAD + band_start; read_pos < band_start + band_widths[score_gIdx]; read_pos += WARP_SIZE * CELLS_PER_THREAD)
        {
            SizeT rIdx        = read_pos / CELLS_PER_THREAD;
            SeqT4<SeqT> read4 = d_read4[rIdx];

            ScoreT4<ScoreT> char_profile;
            char_profile.s0 = (graph_base == read4.r0 ? match_score : mismatch_score);
            char_profile.s1 = (graph_base == read4.r1 ? match_score : mismatch_score);
            char_profile.s2 = (graph_base == read4.r2 ? match_score : mismatch_score);
            char_profile.s3 = (graph_base == read4.r3 ? match_score : mismatch_score);

            ScoreT4<ScoreT> score = get_scores_adaptive(scores, pred_idx, read_pos, band_starts, band_widths, head_indices, max_column, min_score_value, gap_score, char_profile);

            // Perform same score updates as above, but for rest of predecessors.
            for (uint16_t p = 1; p < pred_count; p++)
            {
                SizeT pred_idx2          = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1;
                ScoreT4<ScoreT> scores_4 = get_scores_adaptive(scores, pred_idx2, read_pos, band_starts, band_widths, head_indices, max_column, min_score_value, gap_score, char_profile);

                score.s0 = max(score.s0, scores_4.s0);
                score.s1 = max(score.s1, scores_4.s1);
                score.s2 = max(score.s2, scores_4.s2);
                score.s3 = max(score.s3, scores_4.s3);
            }

            // While there are changes to the horizontal score values, keep updating the matrix.
            // So loop will only run the number of time there are corrections in the matrix.
            // The any_sync warp primitive lets us easily check if any of the threads had an update.
            bool loop = true;

            while (__any_sync(FULL_MASK, loop))
            {
                // To increase instruction level parallelism, we compute the scores
                // in reverse order (score3 first, then score2, then score1, etc).
                // And then check if any of the scores had an update,
                // and if there's an update then we rerun the loop to capture the effects
                // of the change in the next loop.
                loop = false;

                // The shfl_up lets us grab a value from the lane below.
                ScoreT last_score = __shfl_up_sync(FULL_MASK, score.s3, 1);
                if (lane_idx == 0)
                {
                    last_score = first_element_prev_score;
                }

                ScoreT tscore = max(score.s2 + gap_score, score.s3);
                if (tscore > score.s3)
                {
                    score.s3 = tscore;
                    loop     = true;
                }

                tscore = max(score.s1 + gap_score, score.s2);
                if (tscore > score.s2)
                {
                    score.s2 = tscore;
                    loop     = true;
                }

                tscore = max(score.s0 + gap_score, score.s1);
                if (tscore > score.s1)
                {
                    score.s1 = tscore;
                    loop     = true;
                }

                tscore = max(last_score + gap_score, score.s0);
                if (tscore > score.s0)
                {
                    score.s0 = tscore;
                    loop     = true;
                }
            }

            // Copy over the last element score of the last lane into a register of first lane
            // which can be used to compute the first cell of the next warp.
            first_element_prev_score = __shfl_sync(FULL_MASK, score.s3, WARP_SIZE - 1);

            score_index = static_cast<int64_t>(read_pos + 1 - band_start) + head_indices[score_gIdx];

            scores[score_index]      = score.s0;
            scores[score_index + 1L] = score.s1;
            scores[score_index + 2L] = score.s2;
            scores[score_index + 3L] = score.s3;

            __syncwarp();
        }
    }

    SizeT aligned_nodes = 0;
    if (lane_idx == 0)
    {
        // Find location of the maximum score in the matrix.
        SizeT i       = 0;
        SizeT j       = read_length;
        ScoreT mscore = score_type_min_limit;

        for (SizeT idx = 1; idx <= graph_count; idx++)
        {
            if (outgoing_edge_count[graph[idx - 1]] == 0)
            {
                ScoreT s = get_score_adaptive(scores, idx, j, band_starts, band_widths, head_indices, max_column, min_score_value);
                if (mscore < s)
                {
                    mscore = s;
                    i      = idx;
                }
            }
        }

        // Fill in backtrace
        SizeT prev_i = 0;
        SizeT prev_j = 0;

        int32_t loop_count = 0;
        while (!(i == 0 && j == 0) && loop_count < static_cast<int32_t>(read_length + graph_count + 2))
        {
            loop_count++;
            ScoreT scores_ij = get_score_adaptive(scores, i, j, band_starts, band_widths, head_indices, max_column, min_score_value);
            bool pred_found  = false;
            // Check if move is diagonal.
            if (i != 0 && j != 0)
            {

                SizeT node_id     = graph[i - 1];
                ScoreT match_cost = (nodes[node_id] == read[j - 1] ? match_score : mismatch_score);

                uint16_t pred_count = incoming_edge_count[node_id];
                SizeT pred_i        = (pred_count == 0 ? 0 : (node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1));

                if (scores_ij == (get_score_adaptive(scores, pred_i, static_cast<SizeT>(j - 1), band_starts, band_widths, head_indices, max_column, min_score_value) + match_cost))
                {
                    prev_i     = pred_i;
                    prev_j     = j - 1;
                    pred_found = true;
                }

                if (!pred_found)
                {
                    for (uint16_t p = 1; p < pred_count; p++)
                    {
                        pred_i = (node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1);

                        if (scores_ij == (get_score_adaptive(scores, pred_i, static_cast<SizeT>(j - 1), band_starts, band_widths, head_indices, max_column, min_score_value) + match_cost))
                        {
                            prev_i     = pred_i;
                            prev_j     = j - 1;
                            pred_found = true;
                            break;
                        }
                    }
                }
            }

            // Check if move is vertical.
            if (!pred_found && i != 0)
            {
                SizeT node_id       = graph[i - 1];
                uint16_t pred_count = incoming_edge_count[node_id];
                SizeT pred_i        = (pred_count == 0 ? 0 : node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1);

                if (scores_ij == get_score_adaptive(scores, pred_i, j, band_starts, band_widths, head_indices, max_column, min_score_value) + gap_score)
                {
                    prev_i     = pred_i;
                    prev_j     = j;
                    pred_found = true;
                }

                if (!pred_found)
                {
                    for (uint16_t p = 1; p < pred_count; p++)
                    {
                        pred_i = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1;

                        if (scores_ij == get_score_adaptive(scores, pred_i, j, band_starts, band_widths, head_indices, max_column, min_score_value) + gap_score)
                        {
                            prev_i     = pred_i;
                            prev_j     = j;
                            pred_found = true;
                            break;
                        }
                    }
                }
            }

            // Check if move is horizontal.
            if (!pred_found && scores_ij == get_score_adaptive(scores, i, static_cast<SizeT>(j - 1), band_starts, band_widths, head_indices, max_column, min_score_value) + gap_score)
            {
                prev_i     = i;
                prev_j     = j - 1;
                pred_found = true;
            }

            alignment_graph[aligned_nodes] = (i == prev_i ? -1 : graph[i - 1]);
            alignment_read[aligned_nodes]  = (j == prev_j ? -1 : j - 1);
            aligned_nodes++;

            i = prev_i;
            j = prev_j;
        }

        if (loop_count >= (read_length + graph_count + 2))
        {
            aligned_nodes = -1;
        }

#ifdef NW_VERBOSE_PRINT
        printf("aligned nodes %d, loop count %d\n", aligned_nodes, loop_count);
#endif
    }
    aligned_nodes = __shfl_sync(FULL_MASK, aligned_nodes, 0);
    return aligned_nodes;
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
