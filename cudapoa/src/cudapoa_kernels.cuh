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

// Implementation file for CUDA POA kernels.
#pragma once

#include "cudapoa_nw.cuh"
#include "cudapoa_nw_banded.cuh"
#include "cudapoa_nw_adaptive_banded.cuh"
#include "cudapoa_topsort.cuh"
#include "cudapoa_add_alignment.cuh"
#include "cudapoa_generate_consensus.cuh"
#include "cudapoa_generate_msa.cuh"

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/cudapoa/batch.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/**
 * @brief The main kernel that runs the partial order alignment
 *        algorithm.
 *
 * @param[out] consensus_d                  Device buffer for generated consensus
 * @param[in] sequences_d                   Device buffer with sequences for all windows
 * @param[in] base_weights_d                Device buffer with base weights for all windows
 * @param[in] sequence_lengths_d            Device buffer sequence lengths
 * @param[in] window_details_d              Device buffer with structs encapsulating sequence details per window
 * @param[in] total_windows                 Total number of windows to process
 * @param[in] scores_d                      Device scratch space that scores alignment matrix score
 * @param[in] alignment_graph_d             Device scratch space for backtrace alignment of graph
 * @param[in] alignment_read_d              Device scratch space for backtrace alignment of sequence
 * @param[in] nodes_d                       Device scratch space for storing unique nodes in graph
 * @param[in] incoming_edges_d              Device scratch space for storing incoming edges per node
 * @param[in] incoming_edges_count_d        Device scratch space for storing number of incoming edges per node
 * @param[in] outgoing_edges_d              Device scratch space for storing outgoing edges per node
 * @param[in] outgoing_edges_count_d        Device scratch space for storing number of outgoing edges per node
 * @param[in] incoming_edge_w_d             Device scratch space for storing weight of incoming edges
 * @param[in] outgoing_edge_w_d             Device scratch space for storing weight of outgoing edges
 * @param[in] sorted_poa_d                  Device scratch space for storing sorted graph
 * @param[in] node_id_to_pos_d              Device scratch space for mapping node ID to position in graph
 * @graph[in] node_alignments_d             Device scratch space for storing alignment nodes per node in graph
 * @param[in] node_alignment_count_d        Device scratch space for storing number of aligned nodes
 * @param[in] sorted_poa_local_edge_count_d Device scratch space for maintaining edge counts during topological sort
 * @param[in] node_marks_d_                 Device scratch space for storing node marks when running spoa accurate top sort
 * @param[in] check_aligned_nodes_d_        Device scratch space for storing check for aligned nodes
 * @param[in] nodes_to_visit_d_             Device scratch space for storing stack of nodes to be visited in topsort
 * @param[in] node_coverage_counts_d_       Device scratch space for storing coverage of each node in graph.
 * @param[in] gap_score                     Score for inserting gap into alignment
 * @param[in] mismatch_score                Score for finding a mismatch in alignment
 * @param[in] match_score                   Score for finding a match in alignment
 */
template <typename ScoreT, typename SizeT>
__global__ void generatePOAKernel(uint8_t* consensus_d,
                                  uint8_t* sequences_d,
                                  int8_t* base_weights_d,
                                  SizeT* sequence_lengths_d,
                                  genomeworks::cudapoa::WindowDetails* window_details_d,
                                  int32_t total_windows,
                                  ScoreT* scores_d,
                                  SizeT* alignment_graph_d,
                                  SizeT* alignment_read_d,
                                  uint8_t* nodes_d,
                                  SizeT* incoming_edges_d,
                                  uint16_t* incoming_edge_count_d,
                                  SizeT* outgoing_edges_d,
                                  uint16_t* outgoing_edge_count_d,
                                  uint16_t* incoming_edge_w_d,
                                  uint16_t* outgoing_edge_w_d,
                                  SizeT* sorted_poa_d,
                                  SizeT* node_id_to_pos_d,
                                  SizeT* node_alignments_d,
                                  uint16_t* node_alignment_count_d,
                                  uint16_t* sorted_poa_local_edge_count_d,
                                  uint8_t* node_marks_d_,
                                  bool* check_aligned_nodes_d_,
                                  SizeT* nodes_to_visit_d_,
                                  uint16_t* node_coverage_counts_d_,
                                  ScoreT gap_score,
                                  ScoreT mismatch_score,
                                  ScoreT match_score,
                                  uint32_t max_sequences_per_poa,
                                  SizeT* sequence_begin_nodes_ids_d,
                                  uint16_t* outgoing_edges_coverage_d,
                                  uint16_t* outgoing_edges_coverage_count_d,
                                  uint32_t max_nodes_per_graph,
                                  uint32_t scores_matrix_height,
                                  uint32_t scores_matrix_width,
                                  uint32_t max_limit_consensus_size,
                                  int32_t TPB                = 64,
                                  bool adaptive_banded       = false,
                                  bool static_banded         = false,
                                  bool msa                   = false,
                                  uint32_t static_band_width = 256,
                                  BandMode band_mode         = BandMode::full_band)
{
    // shared error indicator within a warp
    bool warp_error = false;

    int32_t nwindows_per_block = TPB / WARP_SIZE;
    int32_t warp_idx           = threadIdx.x / WARP_SIZE;
    int32_t lane_idx           = threadIdx.x % WARP_SIZE;
    int32_t window_idx         = blockIdx.x * nwindows_per_block + warp_idx;

    if (window_idx >= total_windows)
        return;

    // Find the buffer offsets for each thread within the global memory buffers.
    uint8_t* nodes                        = &nodes_d[max_nodes_per_graph * window_idx];
    SizeT* incoming_edges                 = &incoming_edges_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* incoming_edge_count         = &incoming_edge_count_d[window_idx * max_nodes_per_graph];
    SizeT* outgoing_edges                 = &outgoing_edges_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_count         = &outgoing_edge_count_d[window_idx * max_nodes_per_graph];
    uint16_t* incoming_edge_weights       = &incoming_edge_w_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_weights       = &outgoing_edge_w_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    SizeT* sorted_poa                     = &sorted_poa_d[window_idx * max_nodes_per_graph];
    SizeT* node_id_to_pos                 = &node_id_to_pos_d[window_idx * max_nodes_per_graph];
    SizeT* node_alignments                = &node_alignments_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_ALIGNMENTS];
    uint16_t* node_alignment_count        = &node_alignment_count_d[window_idx * max_nodes_per_graph];
    uint16_t* sorted_poa_local_edge_count = &sorted_poa_local_edge_count_d[window_idx * max_nodes_per_graph];

    int32_t scores_width = window_details_d[window_idx].scores_width;

    int64_t scores_offset;
    int64_t banded_score_matrix_size;
    if (static_banded || adaptive_banded)
    {
        banded_score_matrix_size = static_cast<int64_t>(scores_matrix_height) * static_cast<int64_t>(scores_matrix_width);
        scores_offset            = banded_score_matrix_size * static_cast<int64_t>(window_idx);
    }
    else
    {
        scores_offset = static_cast<int64_t>(window_details_d[window_idx].scores_offset) * static_cast<int64_t>(scores_matrix_height);
    }

    ScoreT* scores = &scores_d[scores_offset];

    SizeT* alignment_graph         = &alignment_graph_d[max_nodes_per_graph * window_idx];
    SizeT* alignment_read          = &alignment_read_d[max_nodes_per_graph * window_idx];
    uint16_t* node_coverage_counts = &node_coverage_counts_d_[max_nodes_per_graph * window_idx];

#ifdef SPOA_ACCURATE
    uint8_t* node_marks       = &node_marks_d_[max_nodes_per_graph * window_idx];
    bool* check_aligned_nodes = &check_aligned_nodes_d_[max_nodes_per_graph * window_idx];
    SizeT* nodes_to_visit     = &nodes_to_visit_d_[max_nodes_per_graph * window_idx];
#endif

    SizeT* sequence_lengths = &sequence_lengths_d[window_details_d[window_idx].seq_len_buffer_offset];

    uint32_t num_sequences = window_details_d[window_idx].num_seqs;
    uint8_t* sequence      = &sequences_d[window_details_d[window_idx].seq_starts];
    int8_t* base_weights   = &base_weights_d[window_details_d[window_idx].seq_starts];

    uint8_t* consensus = &consensus_d[window_idx * max_limit_consensus_size];

    SizeT* sequence_begin_nodes_ids         = nullptr;
    uint16_t* outgoing_edges_coverage       = nullptr;
    uint16_t* outgoing_edges_coverage_count = nullptr;

    if (msa)
    {
        sequence_begin_nodes_ids      = &sequence_begin_nodes_ids_d[window_idx * max_sequences_per_poa];
        outgoing_edges_coverage       = &outgoing_edges_coverage_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa];
        outgoing_edges_coverage_count = &outgoing_edges_coverage_count_d[window_idx * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES];
    }

    if (lane_idx == 0)
    {
        // Create backbone for window based on first sequence in window.
        nodes[0]                                     = sequence[0];
        sorted_poa[0]                                = 0;
        incoming_edge_count[0]                       = 0;
        node_alignment_count[0]                      = 0;
        node_id_to_pos[0]                            = 0;
        outgoing_edge_count[sequence_lengths[0] - 1] = 0;
        incoming_edge_weights[0]                     = base_weights[0];
        node_coverage_counts[0]                      = 1;
        if (msa)
        {
            sequence_begin_nodes_ids[0] = 0;
        }

        //Build the rest of the graphs
        for (SizeT nucleotide_idx = 1; nucleotide_idx < sequence_lengths[0]; nucleotide_idx++)
        {
            nodes[nucleotide_idx]                                          = sequence[nucleotide_idx];
            sorted_poa[nucleotide_idx]                                     = nucleotide_idx;
            outgoing_edges[(nucleotide_idx - 1) * CUDAPOA_MAX_NODE_EDGES]  = nucleotide_idx;
            outgoing_edge_count[nucleotide_idx - 1]                        = 1;
            incoming_edges[nucleotide_idx * CUDAPOA_MAX_NODE_EDGES]        = nucleotide_idx - SizeT(1);
            incoming_edge_weights[nucleotide_idx * CUDAPOA_MAX_NODE_EDGES] = base_weights[nucleotide_idx - 1] + base_weights[nucleotide_idx];
            incoming_edge_count[nucleotide_idx]                            = 1;
            node_alignment_count[nucleotide_idx]                           = 0;
            node_id_to_pos[nucleotide_idx]                                 = nucleotide_idx;
            node_coverage_counts[nucleotide_idx]                           = 1;
            if (msa)
            {
                outgoing_edges_coverage[(nucleotide_idx - 1) * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa] = 0;
                outgoing_edges_coverage_count[(nucleotide_idx - 1) * CUDAPOA_MAX_NODE_EDGES]                   = 1;
            }
        }

        // Clear error code for window.
        consensus[0] = CUDAPOA_KERNEL_NOERROR_ENCOUNTERED;
    }

    __syncwarp();

    // Align each subsequent read, add alignment to graph, run topoligical sort.
    for (SizeT s = 1; s < num_sequences; s++)
    {
        SizeT seq_len = sequence_lengths[s];
        sequence += sequence_lengths[s - 1];     // increment the pointer so it is pointing to correct sequence data
        base_weights += sequence_lengths[s - 1]; // increment the pointer so it is pointing to correct sequence data

        if (lane_idx == 0)
        {
            if (sequence_lengths[0] >= max_nodes_per_graph)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = static_cast<uint8_t>(StatusType::node_count_exceeded_maximum_graph_size);
                warp_error   = true;
            }
        }

        warp_error = __shfl_sync(FULL_MASK, warp_error, 0);
        if (warp_error)
        {
            return;
        }

        // Run Needleman-Wunsch alignment between graph and new sequence.
        SizeT alignment_length;

        if (static_banded || adaptive_banded)
        {
            if (adaptive_banded)
            {
                alignment_length = runNeedlemanWunschAdaptiveBanded<uint8_t, ScoreT, SizeT>(nodes,
                                                                                            sorted_poa,
                                                                                            node_id_to_pos,
                                                                                            sequence_lengths[0],
                                                                                            incoming_edge_count,
                                                                                            incoming_edges,
                                                                                            outgoing_edge_count,
                                                                                            sequence,
                                                                                            seq_len,
                                                                                            scores,
                                                                                            banded_score_matrix_size,
                                                                                            alignment_graph,
                                                                                            alignment_read,
                                                                                            static_band_width,
                                                                                            gap_score,
                                                                                            mismatch_score,
                                                                                            match_score,
                                                                                            int8_t{0});

                __syncwarp();

                if (alignment_length < -2)
                {
                    // rerun with extended band-width
                    alignment_length = runNeedlemanWunschAdaptiveBanded<uint8_t, ScoreT, SizeT>(nodes,
                                                                                                sorted_poa,
                                                                                                node_id_to_pos,
                                                                                                sequence_lengths[0],
                                                                                                incoming_edge_count,
                                                                                                incoming_edges,
                                                                                                outgoing_edge_count,
                                                                                                sequence,
                                                                                                seq_len,
                                                                                                scores,
                                                                                                banded_score_matrix_size,
                                                                                                alignment_graph,
                                                                                                alignment_read,
                                                                                                static_band_width,
                                                                                                gap_score,
                                                                                                mismatch_score,
                                                                                                match_score,
                                                                                                static_cast<int8_t>(alignment_length));
                    __syncwarp();
                }
            }
            else
            {
                alignment_length = runNeedlemanWunschBanded<uint8_t, ScoreT, SizeT>(nodes,
                                                                                    sorted_poa,
                                                                                    node_id_to_pos,
                                                                                    sequence_lengths[0],
                                                                                    incoming_edge_count,
                                                                                    incoming_edges,
                                                                                    outgoing_edge_count,
                                                                                    sequence,
                                                                                    seq_len,
                                                                                    scores,
                                                                                    alignment_graph,
                                                                                    alignment_read,
                                                                                    static_band_width,
                                                                                    gap_score,
                                                                                    mismatch_score,
                                                                                    match_score);
                __syncwarp();
            }
        }
        else
        {
            alignment_length = runNeedlemanWunsch<uint8_t, ScoreT, SizeT>(nodes,
                                                                          sorted_poa,
                                                                          node_id_to_pos,
                                                                          sequence_lengths[0],
                                                                          incoming_edge_count,
                                                                          incoming_edges,
                                                                          outgoing_edge_count,
                                                                          outgoing_edges,
                                                                          sequence,
                                                                          seq_len,
                                                                          scores,
                                                                          scores_width,
                                                                          alignment_graph,
                                                                          alignment_read,
                                                                          gap_score,
                                                                          mismatch_score,
                                                                          match_score);
            __syncwarp();
        }

        if (alignment_length == -1)
        {
            if (lane_idx == 0)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = static_cast<uint8_t>(StatusType::loop_count_exceeded_upper_bound);
            }
            return;
        }
        else if (alignment_length == -2)
        {
            if (lane_idx == 0)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = static_cast<uint8_t>(StatusType::exceeded_adaptive_banded_matrix_size);
            }
            return;
        }

        if (lane_idx == 0)
        {

            // Add alignment to graph.
            SizeT new_node_count;
            uint8_t error_code = addAlignmentToGraph(new_node_count,
                                                     nodes, sequence_lengths[0],
                                                     node_alignments, node_alignment_count,
                                                     incoming_edges, incoming_edge_count,
                                                     outgoing_edges, outgoing_edge_count,
                                                     incoming_edge_weights, outgoing_edge_weights,
                                                     alignment_length,
                                                     sorted_poa, alignment_graph,
                                                     sequence, alignment_read,
                                                     node_coverage_counts,
                                                     base_weights,
                                                     (sequence_begin_nodes_ids + s),
                                                     outgoing_edges_coverage,
                                                     outgoing_edges_coverage_count,
                                                     s,
                                                     max_sequences_per_poa,
                                                     max_nodes_per_graph,
                                                     msa);

            if (error_code != 0)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = error_code;
                warp_error   = true;
            }
            else
            {
                sequence_lengths[0] = new_node_count;
                // Run a topsort on the graph.
#ifdef SPOA_ACCURATE
                // Exactly matches racon CPU results
                raconTopologicalSortDeviceUtil(sorted_poa,
                                               node_id_to_pos,
                                               new_node_count,
                                               incoming_edge_count,
                                               incoming_edges,
                                               node_alignment_count,
                                               node_alignments,
                                               node_marks,
                                               check_aligned_nodes,
                                               nodes_to_visit,
                                               (uint16_t)max_nodes_per_graph);
#else
                // Faster top sort
                topologicalSortDeviceUtil(sorted_poa,
                                          node_id_to_pos,
                                          new_node_count,
                                          incoming_edge_count,
                                          outgoing_edges,
                                          outgoing_edge_count,
                                          sorted_poa_local_edge_count);
#endif
            }
        }

        __syncwarp();

        warp_error = __shfl_sync(FULL_MASK, warp_error, 0);
        if (warp_error)
        {
            return;
        }
    }
}

template <typename ScoreT, typename SizeT>
void generatePOA(genomeworks::cudapoa::OutputDetails* output_details_d,
                 genomeworks::cudapoa::InputDetails<SizeT>* input_details_d,
                 int32_t total_windows,
                 cudaStream_t stream,
                 genomeworks::cudapoa::AlignmentDetails<ScoreT, SizeT>* alignment_details_d,
                 genomeworks::cudapoa::GraphDetails<SizeT>* graph_details_d,
                 ScoreT gap_score,
                 ScoreT mismatch_score,
                 ScoreT match_score,
                 bool static_banded,
                 bool adaptive_banded,
                 uint32_t max_sequences_per_poa,
                 int8_t output_mask,
                 const BatchConfig& batch_size)
{
    // unpack output details
    uint8_t* consensus_d                  = output_details_d->consensus;
    uint16_t* coverage_d                  = output_details_d->coverage;
    uint8_t* multiple_sequence_alignments = output_details_d->multiple_sequence_alignments;

    // unpack input details
    uint8_t* sequences_d            = input_details_d->sequences;
    int8_t* base_weights_d          = input_details_d->base_weights;
    SizeT* sequence_lengths_d       = input_details_d->sequence_lengths;
    WindowDetails* window_details_d = input_details_d->window_details;
    SizeT* sequence_begin_nodes_ids = input_details_d->sequence_begin_nodes_ids;

    // unpack alignment details
    ScoreT* scores         = alignment_details_d->scores;
    SizeT* alignment_graph = alignment_details_d->alignment_graph;
    SizeT* alignment_read  = alignment_details_d->alignment_read;

    // unpack graph details
    uint8_t* nodes                          = graph_details_d->nodes;
    SizeT* node_alignments                  = graph_details_d->node_alignments;
    uint16_t* node_alignment_count          = graph_details_d->node_alignment_count;
    SizeT* incoming_edges                   = graph_details_d->incoming_edges;
    uint16_t* incoming_edge_count           = graph_details_d->incoming_edge_count;
    SizeT* outgoing_edges                   = graph_details_d->outgoing_edges;
    uint16_t* outgoing_edge_count           = graph_details_d->outgoing_edge_count;
    uint16_t* incoming_edge_w               = graph_details_d->incoming_edge_weights;
    uint16_t* outgoing_edge_w               = graph_details_d->outgoing_edge_weights;
    SizeT* sorted_poa                       = graph_details_d->sorted_poa;
    SizeT* node_id_to_pos                   = graph_details_d->sorted_poa_node_map;
    uint16_t* sorted_poa_local_edge_count   = graph_details_d->sorted_poa_local_edge_count;
    int32_t* consensus_scores               = graph_details_d->consensus_scores;
    SizeT* consensus_predecessors           = graph_details_d->consensus_predecessors;
    uint8_t* node_marks                     = graph_details_d->node_marks;
    bool* check_aligned_nodes               = graph_details_d->check_aligned_nodes;
    SizeT* nodes_to_visit                   = graph_details_d->nodes_to_visit;
    uint16_t* node_coverage_counts          = graph_details_d->node_coverage_counts;
    uint16_t* outgoing_edges_coverage       = graph_details_d->outgoing_edges_coverage;
    uint16_t* outgoing_edges_coverage_count = graph_details_d->outgoing_edges_coverage_count;
    SizeT* node_id_to_msa_pos               = graph_details_d->node_id_to_msa_pos;

    int32_t nwindows_per_block     = CUDAPOA_THREADS_PER_BLOCK / WARP_SIZE;
    int32_t nblocks                = (static_banded || adaptive_banded) ? total_windows : (total_windows + nwindows_per_block - 1) / nwindows_per_block;
    int32_t TPB                    = (static_banded || adaptive_banded) ? CUDAPOA_BANDED_THREADS_PER_BLOCK : CUDAPOA_THREADS_PER_BLOCK;
    int32_t max_nodes_per_graph    = batch_size.max_nodes_per_graph;
    int32_t matrix_graph_dimension = batch_size.matrix_graph_dimension;
    int32_t matrix_seq_dimension   = batch_size.matrix_sequence_dimension;
    bool msa                       = output_mask & OutputType::msa;

    GW_CU_CHECK_ERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    int32_t consensus_num_blocks = (total_windows / CUDAPOA_MAX_CONSENSUS_PER_BLOCK) + 1;

    generatePOAKernel<ScoreT, SizeT>
        <<<nblocks, TPB, 0, stream>>>(consensus_d,
                                      sequences_d,
                                      base_weights_d,
                                      sequence_lengths_d,
                                      window_details_d,
                                      total_windows,
                                      scores,
                                      alignment_graph,
                                      alignment_read,
                                      nodes,
                                      incoming_edges,
                                      incoming_edge_count,
                                      outgoing_edges,
                                      outgoing_edge_count,
                                      incoming_edge_w,
                                      outgoing_edge_w,
                                      sorted_poa,
                                      node_id_to_pos,
                                      node_alignments,
                                      node_alignment_count,
                                      sorted_poa_local_edge_count,
                                      node_marks,
                                      check_aligned_nodes,
                                      nodes_to_visit,
                                      node_coverage_counts,
                                      gap_score,
                                      mismatch_score,
                                      match_score,
                                      max_sequences_per_poa,
                                      sequence_begin_nodes_ids,
                                      outgoing_edges_coverage,
                                      outgoing_edges_coverage_count,
                                      max_nodes_per_graph,
                                      matrix_graph_dimension,
                                      matrix_seq_dimension,
                                      batch_size.max_consensus_size,
                                      TPB,
                                      adaptive_banded,
                                      static_banded,
                                      msa,
                                      batch_size.alignment_band_width);
    GW_CU_CHECK_ERR(cudaPeekAtLastError());

    if (msa)
    {
        generateMSAKernel<SizeT>
            <<<total_windows, max_sequences_per_poa, 0, stream>>>(nodes,
                                                                  consensus_d,
                                                                  window_details_d,
                                                                  incoming_edge_count,
                                                                  incoming_edges,
                                                                  outgoing_edge_count,
                                                                  outgoing_edges,
                                                                  outgoing_edges_coverage,
                                                                  outgoing_edges_coverage_count,
                                                                  node_id_to_msa_pos,
                                                                  sequence_begin_nodes_ids,
                                                                  multiple_sequence_alignments,
                                                                  sequence_lengths_d,
                                                                  sorted_poa,
                                                                  node_alignments,
                                                                  node_alignment_count,
                                                                  max_sequences_per_poa,
                                                                  node_id_to_pos,
                                                                  node_marks,
                                                                  check_aligned_nodes,
                                                                  nodes_to_visit,
                                                                  max_nodes_per_graph,
                                                                  batch_size.max_consensus_size);
        GW_CU_CHECK_ERR(cudaPeekAtLastError());
    }
    else
    {
        generateConsensusKernel<SizeT>
            <<<consensus_num_blocks, CUDAPOA_MAX_CONSENSUS_PER_BLOCK, 0, stream>>>(consensus_d,
                                                                                   coverage_d,
                                                                                   sequence_lengths_d,
                                                                                   window_details_d,
                                                                                   total_windows,
                                                                                   nodes,
                                                                                   incoming_edges,
                                                                                   incoming_edge_count,
                                                                                   outgoing_edges,
                                                                                   outgoing_edge_count,
                                                                                   incoming_edge_w,
                                                                                   sorted_poa,
                                                                                   node_id_to_pos,
                                                                                   node_alignments,
                                                                                   node_alignment_count,
                                                                                   consensus_scores,
                                                                                   consensus_predecessors,
                                                                                   node_coverage_counts,
                                                                                   max_nodes_per_graph,
                                                                                   batch_size.max_consensus_size);
        GW_CU_CHECK_ERR(cudaPeekAtLastError());
    }
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
