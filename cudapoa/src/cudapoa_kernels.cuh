// Header for for CUDA POA host kernel wrappers.

#pragma once

#include <stdint.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string>
#include <vector>

// Maximum vnumber of edges per node.
#define CUDAPOA_MAX_NODE_EDGES 50

// Maximum number of nodes aligned to each other.
#define CUDAPOA_MAX_NODE_ALIGNMENTS 50

// Maximum number of nodes in a graph, 1 graph per window.
#define CUDAPOA_MAX_NODES_PER_WINDOW 3072

// Maximum number of elements in a sequence.
#define CUDAPOA_MAX_SEQUENCE_SIZE 1024

// Maximum vertical dimension of scoring matrix, which stores graph.
// Adding 4 elements more to ensure a 4byte boundary alignment for
// any allocated buffer.
#define CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION (CUDAPOA_MAX_NODES_PER_WINDOW + 4)
 
// Maximum horizontal dimension of scoring matrix, which stores sequences.
// Adding 4 elements more to ensure a 4byte boundary alignment for
// any allocated buffer.
#define CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION (CUDAPOA_MAX_SEQUENCE_SIZE + 4)


#define CUDAPOA_THREADS_PER_BLOCK 64

#define FULL_MASK 0xffffffff

namespace genomeworks {

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

typedef struct OutputDetails
{
    // Buffer pointer for storing consensus.
    uint8_t* consensus;
    // Buffer for coverage of consensus.
    uint16_t* coverage;
} OutputDetails;

typedef struct InputDetails
{
    // Buffer pointer for input data.
    uint8_t* sequences;
    // Buffer for sequence lengths.
    uint16_t* sequence_lengths;
    // Buffer pointers that hold Window Details struct.
    WindowDetails* window_details;
} InputDetails;

typedef struct AlignmentDetails
{
    // Device buffer for the scoring matrix for all windows.
    int16_t* scores;
    // Device buffers for alignment backtrace
    int16_t* alignment_graph;
    int16_t* alignment_read;
} AlignmentDetails;

typedef struct GraphDetails
{
    // Device buffer to store nodes of the graph. The node itself is the base
    // (A, T, C, G) and the id of the node is it's position in the buffer.
    uint8_t* nodes;

    // Device buffer to store the list of nodes aligned to a 
    // specific node in the graph.
    uint16_t* node_alignments;
    uint16_t* node_alignment_count;

    // Device buffer to store incoming edges to a node.
    uint16_t* incoming_edges;
    uint16_t* incoming_edge_count;

    // Device buffer to store outgoing edges from a node.
    uint16_t* outgoing_edges;
    uint16_t* outgoing_edge_count;

    // Devices buffers to store incoming and outgoing edge weights.
    uint16_t* incoming_edge_weights;
    uint16_t* outgoing_edge_weights;

    // Device buffer to store the topologically sorted graph. Each element
    // of this buffer is an ID of the node.
    uint16_t* sorted_poa;

    // Device buffer that maintains a mapping between the node ID and its
    // position in the topologically sorted graph.
    uint16_t* sorted_poa_node_map;

    // Device buffer used during topological sort to store incoming
    // edge counts for nodes.
    uint16_t* sorted_poa_local_edge_count;

    // Device buffer to store scores calculated during traversal
    // of graph for consensus generation.
    int32_t* consensus_scores;

    // Device buffer to store the predecessors of nodes during
    // graph traversal.
    int16_t* consensus_predecessors;

    // Device buffer to store node marks when performing spoa accurate topsort.
    uint8_t* node_marks;

    // Device buffer to store check for aligned nodes.
    bool* check_aligned_nodes;

    // Device buffer to store stack for nodes to be visited.
    uint16_t* nodes_to_visit;

    // Device buffer for storing coverage of each node in graph.
    uint16_t* node_coverage_counts;
} GraphDetails;


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
 * @param[in] stream                      Stream to run kernel on
 * @param[in] scores                      Device scratch space that scores alignment matrix score
 * @param[in] alignment_graph             Device scratch space for backtrace alignment of graph
 * @param[in] alignment_read              Device scratch space for backtrace alignment of sequence
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
 * @param[in] gap_score                   Score for inserting gap into alignment
 * @param[in] mismatch_score              Score for finding a mismatch in alignment
 * @param[in] match_score                 Score for finding a match in alignment
 */

void generatePOA(genomeworks::cudapoa::OutputDetails * output_details_d,
                 genomeworks::cudapoa::InputDetails * Input_details_d,
                 uint32_t total_windows,
                 cudaStream_t stream,
                 genomeworks::cudapoa::AlignmentDetails * alignment_details_d,
                 genomeworks::cudapoa::GraphDetails * graph_details_d,
                 int16_t gap_score,
                 int16_t mismatch_score,
                 int16_t match_score);


// host function that calls runTopSortKernel
void runTopSort(uint16_t* sorted_poa,
                uint16_t* sorted_poa_node_map,
                uint16_t node_count,
                uint16_t* incoming_edge_count,
                uint16_t* outgoing_edges,
                uint16_t* outgoing_edge_count,
                uint16_t* local_incoming_edge_count);

// Host function that calls the kernel
void addAlignment(uint8_t*  nodes,
                  uint16_t* node_count,
                  uint16_t* node_alignments, uint16_t* node_alignment_count,
                  uint16_t* incoming_edges,  uint16_t* incoming_edge_count,
                  uint16_t* outgoing_edges,  uint16_t* outgoing_edge_count,
                  uint16_t* incoming_edge_w, uint16_t* outgoing_edge_w,
                  uint16_t* alignment_length,
                  uint16_t* graph,
                  int16_t*  alignment_graph,
                  uint8_t*  read,
                  int16_t*  alignment_read,
                  uint16_t* node_coverage_counts);
} // namespace cudapoa

} // namespace genomeworks
