/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
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

template <typename SizeT>
__device__ void set_adaptive_band_arrays(SizeT* node_distance, uint16_t* incoming_edge_count, SizeT* incoming_edges,
                                         SizeT* band_starts, SizeT* band_widths, int64_t* head_indices, SizeT max_row, SizeT max_column)
{
    //    // get M-start and M-end
    //    for (SizeT row_idx = 0; row_idx < max_row; row_idx++)
    //    {
    //        SizeT node_id = graph[row_idx];
    //
    //        uint16_t pred_count = incoming_edge_count[node_id];
    //        if (pred_count == 0)
    //        {
    //        }
    //        else
    //        {
    //            for (uint16_t p = 0; p < pred_count; p++)
    //            {
    //                SizeT pred_node_id        = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
    //                SizeT pred_node_graph_pos = node_id_to_pos[pred_node_id] + 1;
    //                penalty                   = max(penalty, get_score(scores, pred_node_graph_pos, static_cast<SizeT>(0), gradient, band_width, static_cast<SizeT>(read_length + 1), min_score_value));
    //            }
    //            set_score(scores, i, static_cast<SizeT>(0), static_cast<ScoreT>(penalty + gap_score), gradient, band_width, max_column);
    //        }
    //    }

    SizeT dummy_band_width = 256;
    float gradient         = float(max_column) / float(max_row);

    for (SizeT row_idx = 0; row_idx < max_row; row_idx++)
    {
        SizeT start_pos = SizeT(row_idx * gradient) - dummy_band_width / 2;

        start_pos = max(start_pos, 0);

        SizeT end_pos = start_pos + dummy_band_width;

        if (end_pos > max_column)
        {
            start_pos = max_column - dummy_band_width + CELLS_PER_THREAD;
        };

        start_pos = max(start_pos, 0);

        start_pos             = start_pos - (start_pos % CELLS_PER_THREAD);
        band_starts[row_idx]  = start_pos;
        band_widths[row_idx]  = dummy_band_width;
        head_indices[row_idx] = static_cast<int64_t>(row_idx) * (static_cast<int64_t>(dummy_band_width) + static_cast<int64_t>(CUDAPOA_BANDED_MATRIX_RIGHT_PADDING));
    }
}

/**
 * @brief Device function for getting the address of an element specified by (row, column) in the score matrix 
 *        taking adaptive band-width into consideration.
 *        
 * @param[in] scores              Score matrix
 * @param[in] row                 Row # of the element
 * @param[in] column              Column # of the element
 * @param[in] value               Value to set
 * @param[in] band_starts         Array of band_starts per row
 * @param[in] band_widths         Array of band_widths per row
 * @param[in] head_indices      Array of indexes in score that map to band_start per row
 * @param[in] max_column          Last column # in the score matrix
 * @param[out] score_address      Address of the element indicated by the (row, col) tuple
*/

template <typename ScoreT, typename SizeT>
__device__ ScoreT* get_score_ptr_adaptive(ScoreT* scores, SizeT row, SizeT column, SizeT* band_starts, SizeT* band_widths, int64_t* head_indices, SizeT max_column)
{

    SizeT band_start = band_starts[row];

    SizeT col_offset;

    if (column == 0)
    {
        col_offset = 0;
    }
    else
    {
        col_offset = column - band_start;
    }

    int64_t score_index = static_cast<int64_t>(col_offset) + head_indices[row];

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
 * @param[in] band_widths         Array of band_widths per row
 * @param[in] head_indices      Array of indexes in score that map to band_start per row
 * @param[in] max_column          Last column # in the score matrix
*/
template <typename ScoreT, typename SizeT>
__device__ void set_score_adaptive(ScoreT* scores, SizeT row, SizeT column, ScoreT value, SizeT* band_starts, SizeT* band_widths, int64_t* head_indices, SizeT max_column)
{
    SizeT band_start = band_starts[row];

    SizeT col_offset;
    if (column == 0)
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
__device__ void initialize_band_adaptive(ScoreT* scores, SizeT row, ScoreT value, SizeT* band_starts, SizeT* band_widths, int64_t* head_indices, SizeT max_column)
{
    int16_t lane_idx = threadIdx.x % WARP_SIZE;
    SizeT band_start = band_starts[row];
    SizeT band_end   = band_start + band_widths[row];

    SizeT initialization_offset = (band_start == 0) ? 1 : band_start;

    set_score_adaptive(scores, row, initialization_offset, value, band_starts, band_widths, head_indices, max_column);

    for (SizeT j = lane_idx + band_end; j < band_end + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING; j += WARP_SIZE)
    {
        set_score_adaptive(scores, row, j, value, band_starts, band_widths, head_indices, max_column);
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
 * @param[in] head_indices      Array of indexes in score that map to band_start per row
 * @param[in] max_column          Last column # in the score matrix
 * @param[out] score              Score at the specified row and column
*/
template <typename ScoreT, typename SizeT>
__device__ ScoreT get_score_adaptive(ScoreT* scores, SizeT row, SizeT column, SizeT* band_starts, SizeT* band_widths, int64_t* head_indices, SizeT max_column, const ScoreT min_score_value)
{
    SizeT band_start = band_starts[row];
    SizeT band_end   = band_start + band_widths[row];

    if (((column > band_end) || (column < band_start)) && column != 0)
    {
        return min_score_value;
    }
    else
    {
        return *get_score_ptr_adaptive(scores, row, column, band_starts, band_widths, head_indices, max_column);
    }
}

template <typename ScoreT, typename SizeT>
__device__ ScoreT4<ScoreT> get_scores_adaptive(SizeT read_pos,
                                               ScoreT* scores,
                                               SizeT node,
                                               ScoreT gap_score,
                                               ScoreT4<ScoreT> char_profile,
                                               SizeT* band_starts,
                                               SizeT* band_widths,
                                               int64_t* head_indices,
                                               ScoreT default_value,
                                               SizeT max_column)
{

    // The load instructions typically load data in 4B or 8B chunks.
    // If data is 16b (2B), then a 4B load chunk is loaded into register
    // and the necessary bits are extracted before returning. This wastes cycles
    // as each read of 16b issues a separate load command.
    // Instead it is better to load a 4B or 8B chunk into a register
    // using a single load inst, and then extracting necessary part of
    // of the data using bit arithmatic. Also reduces register count.

    SizeT band_start = band_starts[node];

    SizeT band_end = static_cast<SizeT>(band_start + band_widths[node] + CELLS_PER_THREAD);

    if (((static_cast<SizeT>(read_pos + 1) > band_end) || (static_cast<SizeT>(read_pos + 1) < band_start)) && static_cast<SizeT>(read_pos + 1) != 0)
    {
        return ScoreT4<ScoreT>{default_value, default_value, default_value, default_value};
    }
    else
    {
        ScoreT4<ScoreT>* pred_scores = (ScoreT4<ScoreT>*)get_score_ptr_adaptive(scores, node, read_pos, band_starts, band_widths, head_indices, max_column);

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
                       score4_next.s0 + gap_score); // TODO - Do we need to compensate here? @atadkase

        return score;
    }
}

template <typename ScoreT, typename SizeT>
void get_predecessors_max_score_index(SizeT& pred_max_score_left, SizeT& pred_max_score_right,
                                      SizeT node_id, uint16_t* incoming_edge_count, SizeT* incoming_edges, SizeT* node_id_to_pos)
{
    for (uint16_t p = 0; p < incoming_edge_count[node_id]; p++)
    {
        SizeT pred_node_id        = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
        SizeT pred_node_graph_pos = node_id_to_pos[pred_node_id] + 1;
        // find max score index in pred_node_graph_pos

        //ScoreT s = get_score_adaptive(scores, idx, j, band_starts, band_widths, band_head_indices, static_cast<SizeT>(read_length + 1), min_score_value);
    }
}

template <typename ScoreT, typename SizeT>
void get_band_parameters(SizeT node_distance_i, SizeT seq_length)
{
    SizeT pred_max_score_left  = seq_length;
    SizeT pred_max_score_right = 0;
    //get_predecessors_max_score_index(pred_max_score_left, pred_max_score_right);
    SizeT band_start = min(node_distance_i, pred_max_score_left);
    band_start       = band_start < 0 ? 0 : band_start;
    SizeT band_end   = max(node_distance_i, pred_max_score_right);
    band_end         = band_end > seq_length ? seq_length : band_end;
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
                                     SizeT* alignment_graph,
                                     SizeT* alignment_read,
                                     SizeT* node_distance,
                                     SizeT* band_starts,
                                     SizeT* band_widths,
                                     int64_t* head_indices,
                                     SizeT static_band_width,
                                     ScoreT gap_score,
                                     ScoreT mismatch_score,
                                     ScoreT match_score)
{

    GW_CONSTEXPR ScoreT score_type_min_limit = numeric_limits<ScoreT>::min();
    const ScoreT min_score_value             = 2 * abs(min(min(gap_score, mismatch_score), -match_score) - 1) + score_type_min_limit;

    int16_t lane_idx = threadIdx.x % WARP_SIZE;
    int64_t score_index;

    SizeT max_column = read_length + 1;

    if (threadIdx.x == 0)
    {
        set_adaptive_band_arrays(node_distance, incoming_edge_count, incoming_edges, band_starts, band_widths, head_indices, static_cast<SizeT>(graph_count + 1), max_column);
    }

    SizeT max_matrix_sequence_dimension = static_band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING;

    // Initialise the horizontal boundary of the score matrix
    for (SizeT j = lane_idx; j < max_matrix_sequence_dimension; j += WARP_SIZE)
    {
        set_score_adaptive(scores, SizeT{0}, j, static_cast<ScoreT>(j * gap_score), band_starts, band_widths, head_indices, max_column);
    }

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // Initialise the vertical boundary of the score matrix
    if (lane_idx == 0)
    {
#ifdef NW_VERBOSE_PRINT
        printf("graph %d, read %d\n", graph_count, read_length);
#endif

        for (SizeT graph_pos = 0; graph_pos < graph_count; graph_pos++)
        {
            set_score_adaptive(scores, SizeT{0}, SizeT{0}, ScoreT{0}, band_starts, band_widths, head_indices, max_column);

            SizeT node_id = graph[graph_pos];
            SizeT i       = graph_pos + 1;

            uint16_t pred_count = incoming_edge_count[node_id];
            if (pred_count == 0)
            {
                set_score_adaptive(scores, i, SizeT{0}, gap_score, band_starts, band_widths, head_indices, max_column);
            }
            else
            {
                ScoreT penalty = score_type_min_limit;
                for (uint16_t p = 0; p < pred_count; p++)
                {
                    SizeT pred_node_id        = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
                    SizeT pred_node_graph_pos = node_id_to_pos[pred_node_id] + 1;
                    penalty                   = max(penalty, get_score_adaptive(scores, pred_node_graph_pos, SizeT{0}, band_starts, band_widths, head_indices, max_column, min_score_value));
                }
                set_score_adaptive(scores, i, SizeT{0}, static_cast<ScoreT>(penalty + gap_score), band_starts, band_widths, head_indices, max_column);
            }
        }
    }

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    __syncwarp();

    SeqT4<SeqT>* d_read4 = (SeqT4<SeqT>*)read;

    // compute vertical and diagonal values in parallel.
    for (SizeT graph_pos = 0; graph_pos < graph_count; graph_pos++)
    {

        SizeT node_id    = graph[graph_pos];
        SizeT score_gIdx = graph_pos + 1;

        SizeT band_start = band_starts[score_gIdx];

        initialize_band_adaptive(scores, score_gIdx, min_score_value, band_starts, band_widths, head_indices, max_column);

        ScoreT first_element_prev_score = get_score_adaptive(scores, score_gIdx, SizeT{0}, band_starts, band_widths, head_indices, max_column, min_score_value);

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

            ScoreT4<ScoreT> score = get_scores_adaptive(read_pos, scores, pred_idx, gap_score, char_profile, band_starts, band_widths, head_indices, min_score_value, max_column);

            // Perform same score updates as above, but for rest of predecessors.
            for (uint16_t p = 1; p < pred_count; p++)
            {
                SizeT pred_idx2          = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1;
                ScoreT4<ScoreT> scores_4 = get_scores_adaptive(read_pos, scores, pred_idx2, gap_score, char_profile, band_starts, band_widths, head_indices, min_score_value, max_column);

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
