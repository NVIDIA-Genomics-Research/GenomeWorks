/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Implementation file for CUDA POA kernels.

#include "cudapoa_kernels.cuh"
#include "cudapoa_nw.cu"
#include "cudapoa_nw_banded.cu"
#include "cudapoa_topsort.cu"
#include "cudapoa_add_alignment.cu"
#include "cudapoa_generate_consensus.cu"
#include "cudapoa_generate_msa.cu"

#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/cudapoa/batch.hpp>

namespace claragenomics
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
template <int32_t TPB = 64, bool cuda_banded_alignment = false, bool msa = false, typename ScoreT, typename SizeT>
__global__ void generatePOAKernel(uint8_t* consensus_d,
                                  uint8_t* sequences_d,
                                  int8_t* base_weights_d,
                                  SizeT* sequence_lengths_d,
                                  claragenomics::cudapoa::WindowDetails* window_details_d,
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
                                  uint32_t max_limit_nodes_per_window,
                                  uint32_t max_limit_nodes_per_window_banded,
                                  uint32_t max_limit_matrix_graph_dimension,
                                  uint32_t max_limit_matrix_graph_dimension_banded,
                                  uint32_t max_limit_consensus_size)
{
    // shared error indicator within a warp
    bool warp_error = false;

    int32_t nwindows_per_block = TPB / WARP_SIZE;
    int32_t warp_idx           = threadIdx.x / WARP_SIZE;
    int32_t lane_idx           = threadIdx.x % WARP_SIZE;
    int32_t window_idx         = blockIdx.x * nwindows_per_block + warp_idx;

    if (window_idx >= total_windows)
        return;

    // These are not being changed to int32_t to make use of larger range
    // without having to use 2 registers which would be needed for 64bit
    uint32_t max_nodes_per_window = cuda_banded_alignment ? max_limit_nodes_per_window_banded : max_limit_nodes_per_window;
    uint32_t max_graph_dimension  = cuda_banded_alignment ? max_limit_matrix_graph_dimension_banded : max_limit_matrix_graph_dimension;

    // Find the buffer offsets for each thread within the global memory buffers.
    uint8_t* nodes                        = &nodes_d[max_nodes_per_window * window_idx];
    SizeT* incoming_edges                 = &incoming_edges_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* incoming_edge_count         = &incoming_edge_count_d[window_idx * max_nodes_per_window];
    SizeT* outoing_edges                  = &outgoing_edges_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_count         = &outgoing_edge_count_d[window_idx * max_nodes_per_window];
    uint16_t* incoming_edge_weights       = &incoming_edge_w_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_weights       = &outgoing_edge_w_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    SizeT* sorted_poa                     = &sorted_poa_d[window_idx * max_nodes_per_window];
    SizeT* node_id_to_pos                 = &node_id_to_pos_d[window_idx * max_nodes_per_window];
    SizeT* node_alignments                = &node_alignments_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_ALIGNMENTS];
    uint16_t* node_alignment_count        = &node_alignment_count_d[window_idx * max_nodes_per_window];
    uint16_t* sorted_poa_local_edge_count = &sorted_poa_local_edge_count_d[window_idx * max_nodes_per_window];

    int32_t scores_width = window_details_d[window_idx].scores_width;
    size_t scores_offset = window_details_d[window_idx].scores_offset * max_graph_dimension;

    ScoreT* scores = 0;
    if (cuda_banded_alignment)
        scores = &scores_d[max_graph_dimension * CUDAPOA_BANDED_MAX_MATRIX_SEQUENCE_DIMENSION * window_idx];
    else
        scores = &scores_d[scores_offset];

    SizeT* alignment_graph         = &alignment_graph_d[max_graph_dimension * window_idx];
    SizeT* alignment_read          = &alignment_read_d[max_graph_dimension * window_idx];
    uint16_t* node_coverage_counts = &node_coverage_counts_d_[max_nodes_per_window * window_idx];

#ifdef SPOA_ACCURATE
    uint8_t* node_marks       = &node_marks_d_[max_nodes_per_window * window_idx];
    bool* check_aligned_nodes = &check_aligned_nodes_d_[max_nodes_per_window * window_idx];
    SizeT* nodes_to_visit     = &nodes_to_visit_d_[max_nodes_per_window * window_idx];
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
        outgoing_edges_coverage       = &outgoing_edges_coverage_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa];
        outgoing_edges_coverage_count = &outgoing_edges_coverage_count_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
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
            outoing_edges[(nucleotide_idx - 1) * CUDAPOA_MAX_NODE_EDGES]   = nucleotide_idx;
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
    for (uint16_t s = 1; s < num_sequences; s++)
    {
        SizeT seq_len = sequence_lengths[s];
        sequence += sequence_lengths[s - 1];     // increment the pointer so it is pointing to correct sequence data
        base_weights += sequence_lengths[s - 1]; // increment the pointer so it is pointing to correct sequence data

        if (lane_idx == 0)
        {
            if (sequence_lengths[0] >= max_nodes_per_window)
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
        uint16_t alignment_length;

        if (cuda_banded_alignment)
        {
            alignment_length = runNeedlemanWunschBanded<uint8_t, uint16_t, ScoreT>(nodes,
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
                                                                                   gap_score,
                                                                                   mismatch_score,
                                                                                   match_score);
        }
        else
        {
            alignment_length = runNeedlemanWunsch<uint8_t, uint16_t, ScoreT, SizeT>(nodes,
                                                                                    sorted_poa,
                                                                                    node_id_to_pos,
                                                                                    sequence_lengths[0],
                                                                                    incoming_edge_count,
                                                                                    incoming_edges,
                                                                                    outgoing_edge_count,
                                                                                    outoing_edges,
                                                                                    sequence,
                                                                                    seq_len,
                                                                                    scores,
                                                                                    scores_width,
                                                                                    alignment_graph,
                                                                                    alignment_read,
                                                                                    gap_score,
                                                                                    mismatch_score,
                                                                                    match_score);
        }

        __syncwarp();

        if (alignment_length == UINT16_MAX)
        {
            if (lane_idx == 0)
            {
                consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
                consensus[1] = static_cast<uint8_t>(StatusType::loop_count_exceeded_upper_bound);
            }
            return;
        }

        if (lane_idx == 0)
        {

            // Add alignment to graph.
            SizeT new_node_count;
            uint8_t error_code = addAlignmentToGraph<msa>(new_node_count,
                                                          nodes, sequence_lengths[0],
                                                          node_alignments, node_alignment_count,
                                                          incoming_edges, incoming_edge_count,
                                                          outoing_edges, outgoing_edge_count,
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
                                                          max_nodes_per_window);

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
                                               cuda_banded_alignment,
                                               (uint16_t)max_nodes_per_window);
#else
                // Faster top sort
                topologicalSortDeviceUtil(sorted_poa,
                                          node_id_to_pos,
                                          new_node_count,
                                          incoming_edge_count,
                                          outoing_edges,
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
void generatePOAtemplated(claragenomics::cudapoa::OutputDetails* output_details_d,
                          claragenomics::cudapoa::InputDetails<SizeT>* input_details_d,
                          int32_t total_windows,
                          cudaStream_t stream,
                          claragenomics::cudapoa::AlignmentDetails<ScoreT, SizeT>* alignment_details_d,
                          claragenomics::cudapoa::GraphDetails<SizeT>* graph_details_d,
                          ScoreT gap_score,
                          ScoreT mismatch_score,
                          ScoreT match_score,
                          bool cuda_banded_alignment,
                          uint32_t max_sequences_per_poa,
                          int8_t output_mask,
                          const BatchSize& batch_size)
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

    int32_t nwindows_per_block = CUDAPOA_THREADS_PER_BLOCK / WARP_SIZE;
    int32_t nblocks            = (total_windows + nwindows_per_block - 1) / nwindows_per_block;

    CGA_CU_CHECK_ERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    int32_t consensus_num_blocks = (total_windows / CUDAPOA_MAX_CONSENSUS_PER_BLOCK) + 1;
    if (cuda_banded_alignment)
    {
        if (output_mask & OutputType::consensus)
        {
            generatePOAKernel<CUDAPOA_BANDED_THREADS_PER_BLOCK, true, false, ScoreT, SizeT>
                <<<total_windows, CUDAPOA_BANDED_THREADS_PER_BLOCK, 0, stream>>>(consensus_d,
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
                                                                                 batch_size.max_nodes_per_window,
                                                                                 batch_size.max_nodes_per_window_banded,
                                                                                 batch_size.max_matrix_graph_dimension,
                                                                                 batch_size.max_matrix_graph_dimension_banded,
                                                                                 batch_size.max_concensus_size);
            CGA_CU_CHECK_ERR(cudaPeekAtLastError());

            generateConsensusKernel<true, SizeT>
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
                                                                                       batch_size.max_nodes_per_window,
                                                                                       batch_size.max_nodes_per_window_banded,
                                                                                       batch_size.max_concensus_size);
            CGA_CU_CHECK_ERR(cudaPeekAtLastError());
        }
        if (output_mask & OutputType::msa)
        {
            generatePOAKernel<CUDAPOA_BANDED_THREADS_PER_BLOCK, true, true, ScoreT, SizeT>
                <<<total_windows, CUDAPOA_BANDED_THREADS_PER_BLOCK, 0, stream>>>(consensus_d,
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
                                                                                 batch_size.max_nodes_per_window,
                                                                                 batch_size.max_nodes_per_window_banded,
                                                                                 batch_size.max_matrix_graph_dimension,
                                                                                 batch_size.max_matrix_graph_dimension_banded,
                                                                                 batch_size.max_concensus_size);
            CGA_CU_CHECK_ERR(cudaPeekAtLastError());

            generateMSAKernel<true, SizeT>
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
                                                                      batch_size.max_nodes_per_window,
                                                                      batch_size.max_nodes_per_window_banded,
                                                                      batch_size.max_concensus_size);
            CGA_CU_CHECK_ERR(cudaPeekAtLastError());
        }
    }
    else
    {
        if (output_mask & OutputType::consensus)
        {
            generatePOAKernel<CUDAPOA_THREADS_PER_BLOCK, false, false, ScoreT, SizeT>
                <<<nblocks, CUDAPOA_THREADS_PER_BLOCK, 0, stream>>>(consensus_d,
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
                                                                    batch_size.max_nodes_per_window,
                                                                    batch_size.max_nodes_per_window_banded,
                                                                    batch_size.max_matrix_graph_dimension,
                                                                    batch_size.max_matrix_graph_dimension_banded,
                                                                    batch_size.max_concensus_size);
            CGA_CU_CHECK_ERR(cudaPeekAtLastError());

            generateConsensusKernel<false, SizeT>
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
                                                                                       batch_size.max_nodes_per_window,
                                                                                       batch_size.max_nodes_per_window_banded,
                                                                                       batch_size.max_concensus_size);
            CGA_CU_CHECK_ERR(cudaPeekAtLastError());
        }
        if (output_mask & OutputType::msa)
        {
            generatePOAKernel<CUDAPOA_THREADS_PER_BLOCK, false, true, ScoreT, SizeT>
                <<<nblocks, CUDAPOA_THREADS_PER_BLOCK, 0, stream>>>(consensus_d,
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
                                                                    batch_size.max_nodes_per_window,
                                                                    batch_size.max_nodes_per_window_banded,
                                                                    batch_size.max_matrix_graph_dimension,
                                                                    batch_size.max_matrix_graph_dimension_banded,
                                                                    batch_size.max_concensus_size);
            CGA_CU_CHECK_ERR(cudaPeekAtLastError());

            generateMSAKernel<false, SizeT>
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
                                                                      batch_size.max_nodes_per_window,
                                                                      batch_size.max_nodes_per_window_banded,
                                                                      batch_size.max_concensus_size);
            CGA_CU_CHECK_ERR(cudaPeekAtLastError());
        }
    }
}

// Host function call for POA kernel
void generatePOA(claragenomics::cudapoa::OutputDetails* output_details_d,
                 void* input_details_void,
                 int32_t total_windows,
                 cudaStream_t stream,
                 void* alignment_details_void,
                 void* graph_details_void,
                 int16_t gap_score,
                 int16_t mismatch_score,
                 int16_t match_score,
                 bool cuda_banded_alignment,
                 uint32_t max_sequences_per_poa,
                 int8_t output_mask,
                 const BatchSize& batch_size)
{
    if (use32bitScore(batch_size, gap_score, mismatch_score, match_score))
    {
        if (use32bitSize(batch_size, cuda_banded_alignment))
        {
            auto* alignment_details_d = static_cast<claragenomics::cudapoa::AlignmentDetails<int32_t, int32_t>*>(alignment_details_void);
            auto* input_details_d     = static_cast<claragenomics::cudapoa::InputDetails<int32_t>*>(input_details_void);
            auto* graph_details_d     = static_cast<claragenomics::cudapoa::GraphDetails<int32_t>*>(graph_details_void);

            claragenomics::cudapoa::generatePOAtemplated<int32_t, int32_t>(output_details_d,
                                                                           input_details_d,
                                                                           total_windows,
                                                                           stream,
                                                                           alignment_details_d,
                                                                           graph_details_d,
                                                                           (int32_t)gap_score,
                                                                           (int32_t)mismatch_score,
                                                                           (int32_t)match_score,
                                                                           cuda_banded_alignment,
                                                                           max_sequences_per_poa,
                                                                           output_mask,
                                                                           batch_size);
        }
        else
        {
            auto* alignment_details_d = static_cast<claragenomics::cudapoa::AlignmentDetails<int32_t, int16_t>*>(alignment_details_void);
            auto* input_details_d     = static_cast<claragenomics::cudapoa::InputDetails<int16_t>*>(input_details_void);
            auto* graph_details_d     = static_cast<claragenomics::cudapoa::GraphDetails<int16_t>*>(graph_details_void);

            claragenomics::cudapoa::generatePOAtemplated<int32_t, int16_t>(output_details_d,
                                                                           input_details_d,
                                                                           total_windows,
                                                                           stream,
                                                                           alignment_details_d,
                                                                           graph_details_d,
                                                                           (int32_t)gap_score,
                                                                           (int32_t)mismatch_score,
                                                                           (int32_t)match_score,
                                                                           cuda_banded_alignment,
                                                                           max_sequences_per_poa,
                                                                           output_mask,
                                                                           batch_size);
        }
    }
    else
    {
        // if ScoreT is 16-bit, then it's safe to assume SizeT is 16-bit
        auto* alignment_details_d = static_cast<claragenomics::cudapoa::AlignmentDetails<int16_t, int16_t>*>(alignment_details_void);
        auto* input_details_d     = static_cast<claragenomics::cudapoa::InputDetails<int16_t>*>(input_details_void);
        auto* graph_details_d     = static_cast<claragenomics::cudapoa::GraphDetails<int16_t>*>(graph_details_void);

        claragenomics::cudapoa::generatePOAtemplated<int16_t, int16_t>(output_details_d,
                                                                       input_details_d,
                                                                       total_windows,
                                                                       stream,
                                                                       alignment_details_d,
                                                                       graph_details_d,
                                                                       gap_score,
                                                                       mismatch_score,
                                                                       match_score,
                                                                       cuda_banded_alignment,
                                                                       max_sequences_per_poa,
                                                                       output_mask,
                                                                       batch_size);
    }
}

void addAlignment(uint8_t* nodes,
                  void* node_count_void,
                  void* node_alignments_void, uint16_t* node_alignment_count,
                  void* incoming_edges_void, uint16_t* incoming_edge_count,
                  void* outgoing_edges_void, uint16_t* outgoing_edge_count,
                  uint16_t* incoming_edge_w, uint16_t* outgoing_edge_w,
                  uint16_t* alignment_length,
                  void* graph_void,
                  void* alignment_graph_void,
                  uint8_t* read,
                  void* alignment_read_void,
                  uint16_t* node_coverage_counts,
                  int8_t* base_weights,
                  void* sequence_begin_nodes_ids_void,
                  uint16_t* outgoing_edges_coverage,
                  uint16_t* outgoing_edges_coverage_count,
                  uint16_t s,
                  uint32_t max_sequences_per_poa,
                  uint32_t max_limit_nodes_per_window,
                  bool cuda_banded_alignment,
                  const BatchSize& batch_size)
{
    if (use32bitSize(batch_size, cuda_banded_alignment))
    {
        auto* node_count               = static_cast<int32_t*>(node_count_void);
        auto* node_alignments          = static_cast<int32_t*>(node_alignments_void);
        auto* incoming_edges           = static_cast<int32_t*>(incoming_edges_void);
        auto* outgoing_edges           = static_cast<int32_t*>(outgoing_edges_void);
        auto* graph                    = static_cast<int32_t*>(graph_void);
        auto* alignment_graph          = static_cast<int32_t*>(alignment_graph_void);
        auto* alignment_read           = static_cast<int32_t*>(alignment_read_void);
        auto* sequence_begin_nodes_ids = static_cast<int32_t*>(sequence_begin_nodes_ids_void);

        addAlignmentTemplated<int32_t>(nodes,
                                       node_count,
                                       node_alignments, node_alignment_count,
                                       incoming_edges, incoming_edge_count,
                                       outgoing_edges, outgoing_edge_count,
                                       incoming_edge_w, outgoing_edge_w,
                                       alignment_length,
                                       graph,
                                       alignment_graph,
                                       read,
                                       alignment_read,
                                       node_coverage_counts,
                                       base_weights,
                                       sequence_begin_nodes_ids,
                                       outgoing_edges_coverage,
                                       outgoing_edges_coverage_count,
                                       s,
                                       max_sequences_per_poa,
                                       max_limit_nodes_per_window);
    }
    else
    {
        auto* node_count               = static_cast<int16_t*>(node_count_void);
        auto* node_alignments          = static_cast<int16_t*>(node_alignments_void);
        auto* incoming_edges           = static_cast<int16_t*>(incoming_edges_void);
        auto* outgoing_edges           = static_cast<int16_t*>(outgoing_edges_void);
        auto* graph                    = static_cast<int16_t*>(graph_void);
        auto* alignment_graph          = static_cast<int16_t*>(alignment_graph_void);
        auto* alignment_read           = static_cast<int16_t*>(alignment_read_void);
        auto* sequence_begin_nodes_ids = static_cast<int16_t*>(sequence_begin_nodes_ids_void);

        addAlignmentTemplated<int16_t>(nodes,
                                       node_count,
                                       node_alignments, node_alignment_count,
                                       incoming_edges, incoming_edge_count,
                                       outgoing_edges, outgoing_edge_count,
                                       incoming_edge_w, outgoing_edge_w,
                                       alignment_length,
                                       graph,
                                       alignment_graph,
                                       read,
                                       alignment_read,
                                       node_coverage_counts,
                                       base_weights,
                                       sequence_begin_nodes_ids,
                                       outgoing_edges_coverage,
                                       outgoing_edges_coverage_count,
                                       s,
                                       max_sequences_per_poa,
                                       max_limit_nodes_per_window);
    }
}

void runNW(uint8_t* nodes,
           void* graph_void,
           void* node_id_to_pos_void,
           int32_t graph_count,
           uint16_t* incoming_edge_count,
           void* incoming_edges_void,
           uint16_t* outgoing_edge_count,
           void* outgoing_edges_void,
           uint8_t* read,
           uint16_t read_count,
           int16_t* scores,
           int32_t scores_width,
           void* alignment_graph_void,
           void* alignment_read_void,
           int16_t gap_score,
           int16_t mismatch_score,
           int16_t match_score,
           void* aligned_nodes_void,
           bool cuda_banded_alignment,
           const BatchSize& batch_size)
{
    if (use32bitSize(batch_size, cuda_banded_alignment))
    {
        auto* graph           = static_cast<int32_t*>(graph_void);
        auto* node_id_to_pos  = static_cast<int32_t*>(node_id_to_pos_void);
        auto* incoming_edges  = static_cast<int32_t*>(incoming_edges_void);
        auto* outgoing_edges  = static_cast<int32_t*>(outgoing_edges_void);
        auto* alignment_graph = static_cast<int32_t*>(alignment_graph_void);
        auto* alignment_read  = static_cast<int32_t*>(alignment_read_void);
        auto* aligned_nodes   = static_cast<int32_t*>(aligned_nodes_void);

        runNWtemplated<int32_t>(nodes,
                                graph,
                                node_id_to_pos,
                                graph_count,
                                incoming_edge_count,
                                incoming_edges,
                                outgoing_edge_count,
                                outgoing_edges,
                                read,
                                read_count,
                                scores,
                                scores_width,
                                alignment_graph,
                                alignment_read,
                                gap_score,
                                mismatch_score,
                                match_score,
                                aligned_nodes);
    }
    else
    {
        auto* graph           = static_cast<int16_t*>(graph_void);
        auto* node_id_to_pos  = static_cast<int16_t*>(node_id_to_pos_void);
        auto* incoming_edges  = static_cast<int16_t*>(incoming_edges_void);
        auto* outgoing_edges  = static_cast<int16_t*>(outgoing_edges_void);
        auto* alignment_graph = static_cast<int16_t*>(alignment_graph_void);
        auto* alignment_read  = static_cast<int16_t*>(alignment_read_void);
        auto* aligned_nodes   = static_cast<int16_t*>(aligned_nodes_void);

        runNWtemplated<int16_t>(nodes,
                                graph,
                                node_id_to_pos,
                                graph_count,
                                incoming_edge_count,
                                incoming_edges,
                                outgoing_edge_count,
                                outgoing_edges,
                                read,
                                read_count,
                                scores,
                                scores_width,
                                alignment_graph,
                                alignment_read,
                                gap_score,
                                mismatch_score,
                                match_score,
                                aligned_nodes);
    }
}

void generateConsensusTestHost(uint8_t* nodes,
                               int32_t node_count,
                               void* graph_void,
                               void* node_id_to_pos_void,
                               void* incoming_edges_void,
                               uint16_t* incoming_edge_count,
                               void* outgoing_edges_void,
                               uint16_t* outgoing_edge_count,
                               uint16_t* incoming_edge_w,
                               void* predecessors_void,
                               int32_t* scores,
                               uint8_t* consensus,
                               uint16_t* coverage,
                               uint16_t* node_coverage_counts,
                               void* node_alignments_void,
                               uint16_t* node_alignment_count,
                               uint32_t max_limit_consensus_size,
                               bool cuda_banded_alignment,
                               const BatchSize& batch_size)
{
    if (use32bitSize(batch_size, cuda_banded_alignment))
    {
        auto* graph           = static_cast<int32_t*>(graph_void);
        auto* node_id_to_pos  = static_cast<int32_t*>(node_id_to_pos_void);
        auto* incoming_edges  = static_cast<int32_t*>(incoming_edges_void);
        auto* outgoing_edges  = static_cast<int32_t*>(outgoing_edges_void);
        auto* predecessors    = static_cast<int32_t*>(predecessors_void);
        auto* node_alignments = static_cast<int32_t*>(node_alignments_void);

        generateConsensusTemplated(nodes,
                                   node_count,
                                   graph,
                                   node_id_to_pos,
                                   incoming_edges,
                                   incoming_edge_count,
                                   outgoing_edges,
                                   outgoing_edge_count,
                                   incoming_edge_w,
                                   predecessors,
                                   scores,
                                   consensus,
                                   coverage,
                                   node_coverage_counts,
                                   node_alignments,
                                   node_alignment_count,
                                   max_limit_consensus_size);
    }
    else
    {
        auto* graph           = static_cast<int16_t*>(graph_void);
        auto* node_id_to_pos  = static_cast<int16_t*>(node_id_to_pos_void);
        auto* incoming_edges  = static_cast<int16_t*>(incoming_edges_void);
        auto* outgoing_edges  = static_cast<int16_t*>(outgoing_edges_void);
        auto* predecessors    = static_cast<int16_t*>(predecessors_void);
        auto* node_alignments = static_cast<int16_t*>(node_alignments_void);

        generateConsensusTemplated(nodes,
                                   node_count,
                                   graph,
                                   node_id_to_pos,
                                   incoming_edges,
                                   incoming_edge_count,
                                   outgoing_edges,
                                   outgoing_edge_count,
                                   incoming_edge_w,
                                   predecessors,
                                   scores,
                                   consensus,
                                   coverage,
                                   node_coverage_counts,
                                   node_alignments,
                                   node_alignment_count,
                                   max_limit_consensus_size);
    }
}

void runTopSort(void* sorted_poa_void,
                void* sorted_poa_node_map_void,
                int32_t node_count,
                uint16_t* incoming_edge_count,
                void* outgoing_edges_void,
                uint16_t* outgoing_edge_count,
                uint16_t* local_incoming_edge_count,
                bool cuda_banded_alignment,
                const BatchSize& batch_size)
{
    if (use32bitSize(batch_size, cuda_banded_alignment))
    {
        auto* sorted_poa          = static_cast<int32_t*>(sorted_poa_void);
        auto* sorted_poa_node_map = static_cast<int32_t*>(sorted_poa_node_map_void);
        auto* outgoing_edges      = static_cast<int32_t*>(outgoing_edges_void);

        runTopSortTemplated(sorted_poa,
                            sorted_poa_node_map,
                            node_count,
                            incoming_edge_count,
                            outgoing_edges,
                            outgoing_edge_count,
                            local_incoming_edge_count);
    }
    else
    {
        auto* sorted_poa          = static_cast<int16_t*>(sorted_poa_void);
        auto* sorted_poa_node_map = static_cast<int16_t*>(sorted_poa_node_map_void);
        auto* outgoing_edges      = static_cast<int16_t*>(outgoing_edges_void);

        runTopSortTemplated(sorted_poa,
                            sorted_poa_node_map,
                            node_count,
                            incoming_edge_count,
                            outgoing_edges,
                            outgoing_edge_count,
                            local_incoming_edge_count);
    }
}

bool use32bitScore(const BatchSize& batch_size, const int16_t gap_score, const int16_t mismatch_score, const int16_t match_score)
{
    // theoretical max score takes place when sequence and graph completely match with each other
    int32_t upper_bound = batch_size.max_sequence_size * match_score;
    // theoretical min score takes place when sequence and graph do not include a single match
    // it is assumed max_sequence_size <= max_matrix_graph_dimension; gap_score and match_scores are negative, and match_score is positive
    int32_t lower_bound = batch_size.max_sequence_size * std::max(gap_score, mismatch_score) + (batch_size.max_matrix_graph_dimension - batch_size.max_sequence_size) * gap_score;
    // if theoretical upper or lower bound exceed the range represented by int16_t, then int32_t should be used
    return (upper_bound > INT16_MAX || (-lower_bound) > (INT16_MAX + 1));
}

bool use32bitSize(const BatchSize& batch_size, bool banded)
{
    int32_t max_length = batch_size.max_concensus_size;
    if (banded)
    {
        max_length = std::max(max_length, batch_size.max_matrix_graph_dimension_banded);
    }
    else
    {
        max_length = std::max(max_length, batch_size.max_matrix_graph_dimension);
    }
    max_length = std::max(max_length, batch_size.max_matrix_sequence_dimension);
    //if max array length in POA analysis exceeds the range represented by int16_t, then int32_t should be used
    return (max_length > INT16_MAX);
}

} // namespace cudapoa

} // namespace claragenomics
