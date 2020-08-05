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

template <typename SizeT>
__device__ SizeT get_band_start_for_row(SizeT row_idx, float gradient, SizeT band_width, SizeT max_column)
{

    SizeT start_pos = SizeT(row_idx * gradient) - band_width / 2;

    start_pos = max(start_pos, 0);

    SizeT end_pos = start_pos + band_width;

    if (end_pos > max_column)
    {
        start_pos = max_column - band_width + CELLS_PER_THREAD;
    };

    start_pos = max(start_pos, 0);

    start_pos = start_pos - (start_pos % CELLS_PER_THREAD);

    return start_pos;
}

template <typename ScoreT, typename SizeT>
__device__ ScoreT* get_score_ptr(ScoreT* scores, SizeT row, SizeT column, SizeT band_start, SizeT band_width)
{
    column              = column == 0 ? 0 : column - band_start;
    int64_t score_index = static_cast<int64_t>(column) + static_cast<int64_t>(row) * static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
    return &scores[score_index];
};

template <typename ScoreT, typename SizeT>
__device__ void set_score(ScoreT* scores, SizeT row, SizeT column, ScoreT value, float gradient, SizeT band_width, SizeT max_column)
{
    SizeT band_start = get_band_start_for_row(row, gradient, band_width, max_column);

    SizeT col_idx;
    if (column == 0)
    {
        col_idx = band_start;
    }
    else
    {
        col_idx = column - band_start;
    }

    int64_t score_index = static_cast<int64_t>(col_idx) + static_cast<int64_t>(row) * static_cast<int64_t>(band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING);
    scores[score_index] = value;
}

template <typename ScoreT, typename SizeT>
__device__ void initialize_band(ScoreT* scores, SizeT row, ScoreT value, float gradient, SizeT band_width, SizeT max_column)
{
    int16_t lane_idx = threadIdx.x % WARP_SIZE;
    SizeT band_start = get_band_start_for_row(row, gradient, band_width, max_column);
    SizeT band_end   = band_start + band_width;

    SizeT initialization_offset = (band_start == 0) ? 1 : band_start;

    set_score(scores, row, initialization_offset, value, gradient, band_width, max_column);

    // note: as long as CUDAPOA_BANDED_MATRIX_RIGHT_PADDING < WARP_SIZE, no need for a for loop
    for (SizeT j = lane_idx + band_end; j < band_end + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING; j += WARP_SIZE)
    {
        set_score(scores, row, j, value, gradient, band_width, max_column);
    }
};

template <typename ScoreT, typename SizeT>
__device__ ScoreT get_score(ScoreT* scores, SizeT row, SizeT column, float gradient, SizeT band_width, SizeT max_column, const ScoreT min_score_value)
{
    SizeT band_start = get_band_start_for_row(row, gradient, band_width, max_column);
    SizeT band_end   = band_start + band_width;

    if (((column > band_end) || (column < band_start)) && column != 0)
    {
        return min_score_value;
    }
    else
    {
        return *get_score_ptr(scores, row, column, band_start, band_width);
    }
}

template <typename ScoreT, typename SizeT>
__device__ ScoreT4<ScoreT> get_scores(ScoreT* scores,
                                      SizeT node,
                                      SizeT read_pos,
                                      float gradient,
                                      SizeT band_width,
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

    SizeT band_start = get_band_start_for_row(node, gradient, band_width, max_column);

    // subtract by CELLS_PER_THREAD to ensure score4_next is not pointing out of the corresponding band bounds
    SizeT band_end = static_cast<SizeT>(band_start + band_width - CELLS_PER_THREAD);

    if ((read_pos > band_end || read_pos < band_start) && read_pos != 0)
    {
        return ScoreT4<ScoreT>{default_value, default_value, default_value, default_value};
    }
    else
    {
        ScoreT4<ScoreT>* pred_scores = (ScoreT4<ScoreT>*)get_score_ptr(scores, node, read_pos, band_start, band_width);

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

template <typename SeqT,
          typename ScoreT,
          typename SizeT>
__device__
    SizeT
    runNeedlemanWunschBanded(SeqT* nodes,
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
                             SizeT band_width,
                             ScoreT gap_score,
                             ScoreT mismatch_score,
                             ScoreT match_score)
{

    GW_CONSTEXPR ScoreT score_type_min_limit = numeric_limits<ScoreT>::min();
    const ScoreT min_score_value             = 2 * abs(min(min(gap_score, mismatch_score), -match_score) - 1) + score_type_min_limit;

    int16_t lane_idx = threadIdx.x % WARP_SIZE;
    int64_t score_index;

    //Calculate gradient for the scores matrix
    float gradient = float(read_length + 1) / float(graph_count + 1);

    SizeT max_column                    = read_length + 1;
    SizeT max_matrix_sequence_dimension = band_width + CUDAPOA_BANDED_MATRIX_RIGHT_PADDING;
    // Initialise the horizontal boundary of the score matrix
    for (SizeT j = lane_idx; j < max_matrix_sequence_dimension; j += WARP_SIZE)
    {
        set_score(scores, SizeT{0}, j, static_cast<ScoreT>(j * gap_score), gradient, band_width, max_column);
    }

    // Initialise the vertical boundary of the score matrix
    if (lane_idx == 0)
    {
#ifdef NW_VERBOSE_PRINT
        printf("graph %d, read %d\n", graph_count, read_length);
#endif

        for (SizeT graph_pos = 0; graph_pos < graph_count; graph_pos++)
        {
            set_score(scores, SizeT{0}, SizeT{0}, ScoreT{0}, gradient, band_width, max_column);

            SizeT node_id = graph[graph_pos];
            SizeT i       = graph_pos + 1;

            uint16_t pred_count = incoming_edge_count[node_id];
            if (pred_count == 0)
            {
                set_score(scores, i, SizeT{0}, gap_score, gradient, band_width, max_column);
            }
            else
            {
                ScoreT penalty = score_type_min_limit;
                for (uint16_t p = 0; p < pred_count; p++)
                {
                    SizeT pred_node_id        = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
                    SizeT pred_node_graph_pos = node_id_to_pos[pred_node_id] + 1;
                    penalty                   = max(penalty, get_score(scores, pred_node_graph_pos, SizeT{0}, gradient, band_width, max_column, min_score_value));
                }
                set_score(scores, i, SizeT{0}, static_cast<ScoreT>(penalty + gap_score), gradient, band_width, max_column);
            }
        }
    }

    // return;
    __syncwarp();

    SeqT4<SeqT>* d_read4 = (SeqT4<SeqT>*)read;
    // compute vertical and diagonal values in parallel.
    for (SizeT graph_pos = 0; graph_pos < graph_count; graph_pos++)
    {

        SizeT node_id    = graph[graph_pos];
        SizeT score_gIdx = graph_pos + 1;

        SizeT band_start = get_band_start_for_row(score_gIdx, gradient, band_width, max_column);

        initialize_band(scores, score_gIdx, min_score_value, gradient, band_width, max_column);

        ScoreT first_element_prev_score = get_score(scores, score_gIdx, SizeT{0}, gradient, band_width, max_column, min_score_value);

        uint16_t pred_count = incoming_edge_count[node_id];

        SizeT pred_idx = (pred_count == 0 ? 0 : node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1);

        SeqT graph_base = nodes[node_id];

        for (SizeT read_pos = lane_idx * CELLS_PER_THREAD + band_start; read_pos < band_start + band_width; read_pos += WARP_SIZE * CELLS_PER_THREAD)
        {
            SizeT rIdx        = read_pos / CELLS_PER_THREAD;
            SeqT4<SeqT> read4 = d_read4[rIdx];

            ScoreT4<ScoreT> char_profile;
            char_profile.s0 = (graph_base == read4.r0 ? match_score : mismatch_score);
            char_profile.s1 = (graph_base == read4.r1 ? match_score : mismatch_score);
            char_profile.s2 = (graph_base == read4.r2 ? match_score : mismatch_score);
            char_profile.s3 = (graph_base == read4.r3 ? match_score : mismatch_score);

            ScoreT4<ScoreT> score = get_scores(scores, pred_idx, read_pos, gradient, band_width, max_column, min_score_value, gap_score, char_profile);

            // Perform same score updates as above, but for rest of predecessors.
            for (uint16_t p = 1; p < pred_count; p++)
            {
                SizeT pred_idx2          = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1;
                ScoreT4<ScoreT> scores_4 = get_scores(scores, pred_idx2, read_pos, gradient, band_width, max_column, min_score_value, gap_score, char_profile);

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

            score_index = static_cast<int64_t>(read_pos + 1 - band_start) + static_cast<int64_t>(score_gIdx) * static_cast<int64_t>(max_matrix_sequence_dimension);

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
                ScoreT s = get_score(scores, idx, j, gradient, band_width, max_column, min_score_value);
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
            ScoreT scores_ij = get_score(scores, i, j, gradient, band_width, max_column, min_score_value);
            bool pred_found  = false;
            // Check if move is diagonal.
            if (i != 0 && j != 0)
            {

                SizeT node_id     = graph[i - 1];
                ScoreT match_cost = (nodes[node_id] == read[j - 1] ? match_score : mismatch_score);

                uint16_t pred_count = incoming_edge_count[node_id];
                SizeT pred_i        = (pred_count == 0 ? 0 : (node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1));

                if (scores_ij == (get_score(scores, pred_i, static_cast<SizeT>(j - 1), gradient, band_width, max_column, min_score_value) + match_cost))
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

                        if (scores_ij == (get_score(scores, pred_i, static_cast<SizeT>(j - 1), gradient, band_width, max_column, min_score_value) + match_cost))
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

                if (scores_ij == get_score(scores, pred_i, j, gradient, band_width, max_column, min_score_value) + gap_score)
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

                        if (scores_ij == get_score(scores, pred_i, j, gradient, band_width, max_column, min_score_value) + gap_score)
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
            if (!pred_found && scores_ij == get_score(scores, i, static_cast<SizeT>(j - 1), gradient, band_width, max_column, min_score_value) + gap_score)
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
