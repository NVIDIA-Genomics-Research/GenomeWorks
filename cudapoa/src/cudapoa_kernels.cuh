// Header for for CUDA POA host kernel wrappers.

#pragma once

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

// Maximum vnumber of edges per node.
#define CUDAPOA_MAX_NODE_EDGES 50

// Maximum number of nodes aligned to each other.
#define CUDAPOA_MAX_NODE_ALIGNMENTS 50

// Maximum number of nodes in a graph, 1 graph per window.
#define CUDAPOA_MAX_NODES_PER_WINDOW 2048

// Maximum number of elements in a sequence.
#define CUDAPOA_MAX_SEQUENCE_SIZE 1024

// Maximum vertical dimension of scoring matrix, which stores graph.
#define CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION (CUDAPOA_MAX_NODES_PER_WINDOW + 1)

// Maximum horizontal dimension of scoring matrix, which stores sequences.
#define CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION (CUDAPOA_MAX_SEQUENCE_SIZE + 1)

namespace nvidia {

namespace cudapoa {

/**
 * @brief A struct to hold information about the sequences
 *        inside a window.
 */
typedef struct WindowDetails
{
    /// Number of sequences in the window.
    uint16_t num_seqs;
    /// Offset of first sequence length for specific window
    /// inside global sequence length buffer.
    uint32_t seq_len_buffer_offset;
    /// Offset of first sequence content for specific window
    /// inside global sequences buffer.
    uint32_t seq_starts;
} WindowDetails;

/**
 * @brief The host function which calls the kernel that runs the partial order alignment
 *        algorithm.
 *
 * @param[out] consensus_d                Device buffer for generated consensus
 * @param[out] coverage_d_                Device buffer for coverage of each base in consensus
 * @param[in] sequences_d                 Device buffer with sequences for all windows
 * @param[in] sequence_lengths_d          Device buffer sequence lengths
 * @param[in] window_details_d            Device buffer with structs 
 *                                        encapsulating sequence details per window
 * @param[in] total_window                Total number of windows to process
 * @param[in] num_threads                 Number of threads per block
 * @param[in] num_blocks                  Number of blocks to launch
 * @param[in] stream                      Stream to run kernel on
 * @param[in] scores                      Device scratch space that scores alignment matrix score
 * @param[in] ti                          Device scratch space for backtrace alignment of graph
 * @param[in] tj                          Device scratch space for backtrace alignment of sequence
 * @param[in] nodes                       Device scratch space for storing unique nodes in graph
 * @param[in] incoming_edges              Device scratch space for storing incoming edges per node
 * @param[in] incoming_edges_count        Device scratch space for storing number of incoming edges per node
 * @param[in] outgoing_edges              Device scratch space for storing outgoing edges per node
 * @param[in] outgoing_edges_count        Device scratch space for storing number of outgoing edges per node
 * @param[in] incoming_edge_w             Device scratch space for storing weight of incoming edges
 * @param[in] outgoing_edge_w             Device scratch space for storing weight of outgoing edges
 * @param[in] sorted_poa                  Device scratch space for storing sorted graph
 * @param[in] node_id_to_pos              Device scratch space for mapping node ID to position in graph
 * @graph[in] node_alignments             Device scratch space for storing alignment nodes per node in graph
 * @param[in] node_alignment_count        Device scratch space for storing number of aligned nodes
 * @param[in] sorted_poa_local_edge_count Device scratch space for maintaining edge counts during topological sort
 * @param[in] consensus_scores            Device scratch space for storing score of each node while traversing graph during consensus
 * @param[in] consensus_predecessors      Device scratch space for storing predecessors of nodes while traversing graph during consensus
 * @param[in] node_marks_d                Device scratch space for storing node marks when running spoa accurate top sort
 * @param[in] check_aligned_nodes_d       Device scratch space for storing check for aligned nodes
 * @param[in] nodes_to_visit_d            Device scratch space for storing stack of nodes to be visited in topsort
 * @param[in] node_coverage_counts        Device scratch space for storing coverage count for each node in graph
 */
void generatePOA(uint8_t* consensus_d,
                 uint16_t* coverage_d_,
                 uint8_t* sequences_d,
                 uint16_t * sequence_lengths_d,
                 nvidia::cudapoa::WindowDetails * window_details_d,
                 uint32_t total_windows,
                 uint32_t num_threads,
                 uint32_t num_blocks,
                 cudaStream_t stream,
                 int16_t* scores,
                 int16_t* ti,
                 int16_t* tj,
                 uint8_t* nodes,
                 uint16_t* incoming_edges,
                 uint16_t* incoming_edge_count,
                 uint16_t* outgoing_edges,
                 uint16_t* outgoing_edge_count,
                 uint16_t* incoming_edge_w,
                 uint16_t* outgoing_edge_w,
                 uint16_t* sorted_poa,
                 uint16_t* node_id_to_pos,
                 uint16_t* node_alignments,
                 uint16_t* node_alignment_count,
                 uint16_t* sorted_poa_local_edge_count,
                 int32_t* consensus_scores,
                 int16_t* consensus_predecessors,
                 uint8_t* node_marks,
                 bool* check_aligned_nodes,
                 uint16_t* nodes_to_visit,
                 uint16_t* node_coverage_counts);

}

}
