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

template <typename ScoreT>
__device__ __forceinline__
    ScoreT4<ScoreT>
    make_ScoreT4(ScoreT s0)
{
    ScoreT4<ScoreT> t;
    t.s0 = s0;
    t.s1 = s0;
    t.s2 = s0;
    t.s3 = s0;
    return t;
}

template <typename SeqT,
          typename ScoreT,
          typename SizeT>
__device__ __forceinline__
    ScoreT4<ScoreT>
    computeScore(SizeT rIdx,
                 SeqT4<SeqT> read4,
                 SizeT gIdx,
                 SeqT graph_base,
                 uint16_t pred_count,
                 SizeT pred_idx,
                 SizeT* node_id_to_pos,
                 SizeT* incoming_edges,
                 ScoreT* scores,
                 int32_t scores_width,
                 ScoreT gap_score,
                 ScoreT match_score,
                 ScoreT mismatch_score)
{

    ScoreT4<ScoreT> char_profile;
    char_profile.s0 = (graph_base == read4.r0 ? match_score : mismatch_score);
    char_profile.s1 = (graph_base == read4.r1 ? match_score : mismatch_score);
    char_profile.s2 = (graph_base == read4.r2 ? match_score : mismatch_score);
    char_profile.s3 = (graph_base == read4.r3 ? match_score : mismatch_score);

    // The load instructions typically load data in 4B or 8B chunks.
    // If data is 16b (2B), then a 4B load chunk is loaded into register
    // and the necessary bits are extracted before returning. This wastes cycles
    // as each read of 16b issues a separate load command.
    // Instead it is better to load a 4B or 8B chunk into a register
    // using a single load inst, and then extracting necessary part of
    // of the data using bit arithmatic. Also reduces register count.
    int64_t score_index          = static_cast<int64_t>(pred_idx) * static_cast<int64_t>(scores_width);
    ScoreT4<ScoreT>* pred_scores = (ScoreT4<ScoreT>*)&scores[score_index];

    // loads 8 consecutive bytes (4 shorts)
    ScoreT4<ScoreT> score4 = pred_scores[rIdx];

    // need to load the next chunk of memory as well
    ScoreT4<ScoreT> score4_next = pred_scores[rIdx + 1];

    ScoreT4<ScoreT> score;

    score.s0 = max(score4.s0 + char_profile.s0,
                   score4.s1 + gap_score);
    score.s1 = max(score4.s1 + char_profile.s1,
                   score4.s2 + gap_score);
    score.s2 = max(score4.s2 + char_profile.s2,
                   score4.s3 + gap_score);
    score.s3 = max(score4.s3 + char_profile.s3,
                   score4_next.s0 + gap_score);

    // Perform same score updates as above, but for rest of predecessors.
    for (SizeT p = 1; p < pred_count; p++)
    {
        SizeT pred_idx = node_id_to_pos[incoming_edges[gIdx * CUDAPOA_MAX_NODE_EDGES + p]] + 1;

        score_index                  = static_cast<int64_t>(pred_idx) * static_cast<int64_t>(scores_width);
        ScoreT4<ScoreT>* pred_scores = (ScoreT4<ScoreT>*)&scores[score_index];

        // Reasoning for 8B preload same as above.
        ScoreT4<ScoreT> score4      = pred_scores[rIdx];
        ScoreT4<ScoreT> score4_next = pred_scores[rIdx + 1];

        score.s0 = max(score4.s0 + char_profile.s0,
                       max(score.s0, score4.s1 + gap_score));

        score.s1 = max(score4.s1 + char_profile.s1,
                       max(score.s1, score4.s2 + gap_score));

        score.s2 = max(score4.s2 + char_profile.s2,
                       max(score.s2, score4.s3 + gap_score));

        score.s3 = max(score4.s3 + char_profile.s3,
                       max(score.s3, score4_next.s0 + gap_score));
    }

    return score;
}

/**
 * @brief Device function for running Needleman-Wunsch dynamic programming loop.
 *
 * @param[in] nodes                Device buffer with unique nodes in graph
 * @param[in] graph                Device buffer with sorted graph
 * @param[in] node_id_to_pos       Device scratch space for mapping node ID to position in graph
 * @param[in] incoming_edge_count  Device buffer with number of incoming edges per node
 * @param[in] incoming_edges       Device buffer with incoming edges per node
 * @param[in] outgoing_edge_count  Device buffer with number of outgoing edges per node
 * @param[in] outgoing_edges       Device buffer with outgoing edges per node
 * @param[in] read                 Device buffer with sequence (read) to align
 * @param[in] read_length          Number of bases in read
 * @param[out] scores              Device scratch space that scores alignment matrix score
 * @param[out] alignment_graph     Device scratch space for backtrace alignment of graph
 * @param[out] alignment_read      Device scratch space for backtrace alignment of sequence
 * @param[in] gap_score            Score for inserting gap into alignment
 * @param[in] mismatch_score       Score for finding a mismatch in alignment
 * @param[in] match_score          Score for finding a match in alignment
 *
 * @return Number of nodes in final alignment.
 */
template <typename SeqT,
          typename ScoreT,
          typename SizeT,
          int32_t CPT = 4>
__device__
    SizeT
    runNeedlemanWunsch(SeqT* nodes,
                       SizeT* graph,
                       SizeT* node_id_to_pos,
                       SizeT graph_count,
                       uint16_t* incoming_edge_count,
                       SizeT* incoming_edges,
                       uint16_t* outgoing_edge_count,
                       SizeT* outgoing_edges,
                       SeqT* read,
                       SizeT read_length,
                       ScoreT* scores,
                       int32_t scores_width,
                       SizeT* alignment_graph,
                       SizeT* alignment_read,
                       ScoreT gap_score,
                       ScoreT mismatch_score,
                       ScoreT match_score)
{

    static_assert(CPT == 4, "implementation currently supports only 4 cells per thread");

    GW_CONSTEXPR ScoreT score_type_min_limit = numeric_limits<ScoreT>::min();

    int16_t lane_idx = threadIdx.x % WARP_SIZE;
    int64_t score_index;

    // Init horizonal boundary conditions (read).
    for (SizeT j = lane_idx; j < read_length + 1; j += WARP_SIZE)
    {
        scores[j] = j * gap_score;
    }

    if (lane_idx == 0)
    {
#ifdef NW_VERBOSE_PRINT
        printf("graph %d, read %d\n", graph_count, read_length);
#endif
        // Init vertical boundary (graph).
        for (SizeT graph_pos = 0; graph_pos < graph_count; graph_pos++)
        {
            SizeT node_id       = graph[graph_pos];
            SizeT i             = graph_pos + 1;
            uint16_t pred_count = incoming_edge_count[node_id];
            if (pred_count == 0)
            {
                score_index         = static_cast<int64_t>(i) * static_cast<int64_t>(scores_width);
                scores[score_index] = gap_score;
            }
            else
            {
                ScoreT penalty = score_type_min_limit;
                for (uint16_t p = 0; p < pred_count; p++)
                {
                    SizeT pred_node_id        = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
                    SizeT pred_node_graph_pos = node_id_to_pos[pred_node_id] + 1;
                    score_index               = static_cast<int64_t>(pred_node_graph_pos) * static_cast<int64_t>(scores_width);
                    penalty                   = max(penalty, scores[score_index]);
                }
                score_index         = static_cast<int64_t>(i) * static_cast<int64_t>(scores_width);
                scores[score_index] = penalty + gap_score;
            }
        }
    }

    __syncwarp();

    // readpos_bound is the first multiple of (CPT * WARP_SIZE) that is larger than read_length.
    SizeT readpos_bound = (((read_length - 1) / (WARP_SIZE * CPT)) + 1) * (WARP_SIZE * CPT);

    SeqT4<SeqT>* d_read4 = (SeqT4<SeqT>*)read;

    // Run DP loop for calculating scores. Process each row at a time, and
    // compute vertical and diagonal values in parallel.
    for (SizeT graph_pos = 0; graph_pos < graph_count; graph_pos++)
    {

        SizeT node_id    = graph[graph_pos]; // node id for the graph node
        SizeT score_gIdx = graph_pos + 1;    // score matrix index for this graph node

        score_index                     = static_cast<int64_t>(score_gIdx) * static_cast<int64_t>(scores_width);
        ScoreT first_element_prev_score = scores[score_index];

        uint16_t pred_count = incoming_edge_count[node_id];

        SizeT pred_idx = (pred_count == 0 ? 0 : node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1);

        SeqT graph_base = nodes[node_id];

        // readpos_bound is the first tb boundary multiple beyond read_length. This is done
        // so all threads in the block enter the loop. The loop has syncwarp, so if
        // any of the threads don't enter, then it'll cause a lock in the system.
        for (SizeT read_pos = lane_idx * CPT; read_pos < readpos_bound; read_pos += WARP_SIZE * CPT)
        {

            SizeT rIdx = read_pos / CPT;

            // To avoid doing extra work, we clip the extra warps that go beyond the read count.
            // Warp clipping hasn't shown to help too much yet, but might if we increase the tb
            // size in the future.

            SeqT4<SeqT> read4 = d_read4[rIdx];

            ScoreT4<ScoreT> score = make_ScoreT4((ScoreT)SHRT_MAX);

            if (read_pos < read_length)
            {
                score = computeScore<SeqT, ScoreT, SizeT>(rIdx, read4,
                                                          node_id, graph_base,
                                                          pred_count, pred_idx,
                                                          node_id_to_pos, incoming_edges,
                                                          scores, scores_width,
                                                          gap_score, match_score, mismatch_score);
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

            // Index into score matrix.
            score_index = static_cast<int64_t>(score_gIdx) * static_cast<int64_t>(scores_width) + static_cast<int64_t>(read_pos);
            if (read_pos < read_length)
            {
                scores[score_index + 1L] = score.s0;
                scores[score_index + 2L] = score.s1;
                scores[score_index + 3L] = score.s2;
                scores[score_index + 4L] = score.s3;
            }
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
                score_index = static_cast<int64_t>(idx) * static_cast<int64_t>(scores_width) + static_cast<int64_t>(j);
                ScoreT s    = scores[score_index];
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

        // Trace back from maximum score position to generate alignment.
        // Trace back is done by re-calculating the score at each cell
        // along the path to see which preceding cell the move could have
        // come from. This seems computaitonally more expensive, but doesn't
        // require storing any traceback buffer during alignment.
        int32_t loop_count = 0;
        while (!(i == 0 && j == 0) && loop_count < static_cast<int32_t>(read_length + graph_count + 2))
        {
            loop_count++;
            score_index      = static_cast<int64_t>(i) * static_cast<int64_t>(scores_width) + static_cast<int64_t>(j);
            ScoreT scores_ij = scores[score_index];
            bool pred_found  = false;

            // Check if move is diagonal.
            if (i != 0 && j != 0)
            {
                SizeT node_id       = graph[i - 1];
                ScoreT match_cost   = (nodes[node_id] == read[j - 1] ? match_score : mismatch_score);
                uint16_t pred_count = incoming_edge_count[node_id];
                SizeT pred_i        = (pred_count == 0 ? 0 : (node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1));

                score_index = static_cast<int64_t>(pred_i) * static_cast<int64_t>(scores_width) + static_cast<int64_t>(j - 1);
                if (scores_ij == (scores[score_index] + match_cost))
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

                        score_index = static_cast<int64_t>(pred_i) * static_cast<int64_t>(scores_width) + static_cast<int64_t>(j - 1);
                        if (scores_ij == (scores[score_index] + match_cost))
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

                score_index = static_cast<int64_t>(pred_i) * static_cast<int64_t>(scores_width) + static_cast<int64_t>(j);
                if (scores_ij == scores[score_index] + gap_score)
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

                        score_index = static_cast<int64_t>(pred_i) * static_cast<int64_t>(scores_width) + static_cast<int64_t>(j);
                        if (scores_ij == scores[score_index] + gap_score)
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
            score_index = static_cast<int64_t>(i) * static_cast<int64_t>(scores_width) + static_cast<int64_t>(j - 1);
            if (!pred_found && scores_ij == scores[score_index] + gap_score)
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

        } // end of while

        if (loop_count >= (read_length + graph_count + 2))
        {
            aligned_nodes = -1;
        }

#ifdef NW_VERBOSE_PRINT
        printf("aligned nodes %d\n", aligned_nodes);
#endif
    }

    aligned_nodes = __shfl_sync(0xffffffff, aligned_nodes, 0);
    return aligned_nodes;
}

template <typename SizeT>
__global__ void runNeedlemanWunschKernel(uint8_t* nodes,
                                         SizeT* graph,
                                         SizeT* node_id_to_pos,
                                         SizeT graph_count,
                                         uint16_t* incoming_edge_count,
                                         SizeT* incoming_edges,
                                         uint16_t* outgoing_edge_count,
                                         SizeT* outgoing_edges,
                                         uint8_t* read,
                                         SizeT read_length,
                                         int16_t* scores,
                                         int32_t scores_width,
                                         SizeT* alignment_graph,
                                         SizeT* alignment_read,
                                         int16_t gap_score,
                                         int16_t mismatch_score,
                                         int16_t match_score,
                                         SizeT* aligned_nodes)
{
    *aligned_nodes = runNeedlemanWunsch<uint8_t, int16_t, SizeT>(nodes,
                                                                 graph,
                                                                 node_id_to_pos,
                                                                 graph_count,
                                                                 incoming_edge_count,
                                                                 incoming_edges,
                                                                 outgoing_edge_count,
                                                                 outgoing_edges,
                                                                 read,
                                                                 read_length,
                                                                 scores,
                                                                 scores_width,
                                                                 alignment_graph,
                                                                 alignment_read,
                                                                 gap_score,
                                                                 mismatch_score,
                                                                 match_score);
}

// Host function that calls the kernel
template <typename SizeT>
void runNW(uint8_t* nodes,
           SizeT* graph,
           SizeT* node_id_to_pos,
           SizeT graph_count,
           uint16_t* incoming_edge_count,
           SizeT* incoming_edges,
           uint16_t* outgoing_edge_count,
           SizeT* outgoing_edges,
           uint8_t* read,
           SizeT read_length,
           int16_t* scores,
           int32_t scores_width,
           SizeT* alignment_graph,
           SizeT* alignment_read,
           int16_t gap_score,
           int16_t mismatch_score,
           int16_t match_score,
           SizeT* aligned_nodes)
{
    runNeedlemanWunschKernel<<<1, 64>>>(nodes,
                                        graph,
                                        node_id_to_pos,
                                        graph_count,
                                        incoming_edge_count,
                                        incoming_edges,
                                        outgoing_edge_count,
                                        outgoing_edges,
                                        read,
                                        read_length,
                                        scores,
                                        scores_width,
                                        alignment_graph,
                                        alignment_read,
                                        gap_score,
                                        mismatch_score,
                                        match_score,
                                        aligned_nodes);
    GW_CU_CHECK_ERR(cudaPeekAtLastError());
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
