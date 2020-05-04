/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// Header for for CUDA POA host kernel wrappers.

#pragma once

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <claragenomics/cudapoa/batch.hpp>

namespace claragenomics
{

namespace cudapoa
{

/**
 * @brief The host function which calls the kernel that runs the partial order alignment
 *        algorithm.
 *
 * @param[out] output_details_d           Struct that contains output buffers, including the following fields:
 *             consensus                  Device buffer for generated consensus
 *             coverage                   Device buffer for coverage of each base in consensus
 *
 * @param[in] input_details_d             Struct that contains input buffers, including the following fields:
 *            sequences                   Device buffer with sequences for all windows
 *            base_weight                 Device buffer with weights per base for all windows
 *            sequence_lengths            Device buffer sequence lengths
 *            window_details              Device buffer with structs encapsulating sequence details per window
 *
 * @param[in] total_window                Total number of windows to process
 * @param[in] stream                      Stream to run kernel on
 *
 * @param[in] alignment_details_d         Struct that contains alignment related buffers, including the following fields:
 *            scores                      Device scratch space that scores alignment matrix score
 *            alignment_graph             Device scratch space for backtrace alignment of graph
 *            alignment_read              Device scratch space for backtrace alignment of sequence
 *
 * @param[in] graph_details_d             Struct that contains graph related buffers, including the following fields:
 *            nodes                       Device scratch space for storing unique nodes in graph
 *            incoming_edges              Device scratch space for storing incoming edges per node
 *            incoming_edges_count        Device scratch space for storing number of incoming edges per node
 *            outgoing_edges              Device scratch space for storing outgoing edges per node
 *            outgoing_edges_count        Device scratch space for storing number of outgoing edges per node
 *            incoming_edge_w             Device scratch space for storing weight of incoming edges
 *            outgoing_edge_w             Device scratch space for storing weight of outgoing edges
 *            sorted_poa                  Device scratch space for storing sorted graph
 *            node_id_to_pos              Device scratch space for mapping node ID to position in graph
 *            node_alignments             Device scratch space for storing alignment nodes per node in graph
 *            node_alignment_count        Device scratch space for storing number of aligned nodes
 *            sorted_poa_local_edge_count Device scratch space for maintaining edge counts during topological sort
 *            consensus_scores            Device scratch space for storing score of each node while traversing graph during consensus
 *            consensus_predecessors      Device scratch space for storing predecessors of nodes while traversing graph during consensus
 *            node_marks                  Device scratch space for storing node marks when running spoa accurate top sort
 *            check_aligned_nodes         Device scratch space for storing check for aligned nodes
 *            nodes_to_visit              Device scratch space for storing stack of nodes to be visited in topsort
 *            node_coverage_counts        Device scratch space for storing coverage count for each node in graph
 *
 * @param[in] gap_score                   Score for inserting gap into alignment
 * @param[in] mismatch_score              Score for finding a mismatch in alignment
 * @param[in] match_score                 Score for finding a match in alignment
 * @param[in] banded_alignment            Use banded alignment
 */

//void generatePOA(claragenomics::cudapoa::OutputDetails* output_details_d,
//                 void* Input_details_d,
//                 int32_t total_windows,
//                 cudaStream_t stream,
//                 void* alignment_details_d,
//                 void* graph_details_d,
//                 int16_t gap_score,
//                 int16_t mismatch_score,
//                 int16_t match_score,
//                 bool banded_alignment,
//                 uint32_t max_sequences_per_poa,
//                 int8_t output_mask,
//                 const BatchSize& batch_size);
//
//void addAlignment(uint8_t* nodes,
//                  void* node_count_void,
//                  void* node_alignments_void, uint16_t* node_alignment_count,
//                  void* incoming_edges_void, uint16_t* incoming_edge_count,
//                  void* outgoing_edges_void, uint16_t* outgoing_edge_count,
//                  uint16_t* incoming_edge_w, uint16_t* outgoing_edge_w,
//                  uint16_t* alignment_length,
//                  void* graph_void,
//                  void* alignment_graph_void,
//                  uint8_t* read,
//                  void* alignment_read_void,
//                  uint16_t* node_coverage_counts,
//                  int8_t* base_weights,
//                  void* sequence_begin_nodes_ids_void,
//                  uint16_t* outgoing_edges_coverage,
//                  uint16_t* outgoing_edges_coverage_count,
//                  uint16_t s,
//                  uint32_t max_sequences_per_poa,
//                  uint32_t max_limit_nodes_per_window,
//                  bool cuda_banded_alignment,
//                  const BatchSize& batch_size);
//
//void runNW(uint8_t* nodes,
//           void* graph_void,
//           void* node_id_to_pos_void,
//           int32_t graph_count_void,
//           uint16_t* incoming_edge_count,
//           void* incoming_edges_void,
//           uint16_t* outgoing_edge_count,
//           void* outgoing_edges_void,
//           uint8_t* read,
//           uint16_t read_count,
//           int16_t* scores,
//           int32_t scores_width,
//           void* alignment_graph_void,
//           void* alignment_read_void,
//           int16_t gap_score,
//           int16_t mismatch_score,
//           int16_t match_score,
//           void* aligned_nodes_void,
//           bool cuda_banded_alignment,
//           const BatchSize& batch_size);
//
//void generateConsensusTestHost(uint8_t* nodes,
//                               int32_t node_count,
//                               void* graph,
//                               void* node_id_to_pos,
//                               void* incoming_edges,
//                               uint16_t* incoming_edge_count,
//                               void* outgoing_edges,
//                               uint16_t* outgoing_edge_count,
//                               uint16_t* incoming_edge_w,
//                               void* predecessors,
//                               int32_t* scores,
//                               uint8_t* consensus,
//                               uint16_t* coverage,
//                               uint16_t* node_coverage_counts,
//                               void* node_alignments,
//                               uint16_t* node_alignment_count,
//                               uint32_t max_limit_consensus_size,
//                               bool cuda_banded_alignment,
//                               const BatchSize& batch_size);
//
//void runTopSort(void* sorted_poa,
//                void* sorted_poa_node_map,
//                int32_t node_count,
//                uint16_t* incoming_edge_count,
//                void* outgoing_edges,
//                uint16_t* outgoing_edge_count,
//                uint16_t* local_incoming_edge_count,
//                bool cuda_banded_alignment,
//                const BatchSize& batch_size);
//
//// determine proper type definition for ScoreT, used for values of score matrix
//bool use32bitScore(const BatchSize& batch_size, const int16_t gap_score, const int16_t mismatch_score, const int16_t match_score);
//
//// determine proper type definition for SizeT, used for length of arrays in POA
//bool use32bitSize(const BatchSize& batch_size, bool banded);

} // namespace cudapoa

} // namespace claragenomics

#include "cudapoa_kernels.cu"
