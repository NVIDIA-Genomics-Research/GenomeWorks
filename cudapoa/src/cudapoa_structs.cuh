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

#include <stdint.h>

#include <stdio.h>

// Maximum number of edges per node.
#define CUDAPOA_MAX_NODE_EDGES 50

// Maximum number of nodes aligned to each other.
#define CUDAPOA_MAX_NODE_ALIGNMENTS 50

// Dimensions for Banded alignment score matrix
#define WARP_SIZE 32
#define CELLS_PER_THREAD 4
#define CUDAPOA_MIN_BAND_WIDTH (CELLS_PER_THREAD * WARP_SIZE)
#define CUDAPOA_BANDED_MATRIX_RIGHT_PADDING (CELLS_PER_THREAD * 2)

#define CUDAPOA_THREADS_PER_BLOCK 64
#define CUDAPOA_BANDED_THREADS_PER_BLOCK WARP_SIZE
#define CUDAPOA_MAX_CONSENSUS_PER_BLOCK 512

#define FULL_MASK 0xffffffff
#define CUDAPOA_KERNEL_ERROR_ENCOUNTERED UINT8_MAX
#define CUDAPOA_KERNEL_NOERROR_ENCOUNTERED 0

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

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
    int32_t seq_len_buffer_offset;
    /// Offset of first sequence content for specific window
    /// inside global sequences buffer.
    int32_t seq_starts;

    /// Offset to the scores buffer for specific window
    size_t scores_offset;

    /// Max column width of the score matrix required for specific window
    int32_t scores_width;

} WindowDetails;

typedef struct OutputDetails
{
    // Buffer pointer for storing consensus.
    uint8_t* consensus;
    // Buffer for coverage of consensus.
    uint16_t* coverage;
    // Buffer for multiple sequence alignments
    uint8_t* multiple_sequence_alignments;
} OutputDetails;

template <typename SizeT>
struct InputDetails
{
    // Buffer pointer for input data.
    uint8_t* sequences;
    // Buffer pointer for weights of each base.
    int8_t* base_weights;
    // Buffer for sequence lengths.
    SizeT* sequence_lengths;
    // Buffer pointers that hold Window Details struct.
    WindowDetails* window_details;
    // Buffer storing begining nodes for sequences
    SizeT* sequence_begin_nodes_ids;
};

template <typename ScoreT, typename SizeT>
struct AlignmentDetails
{
    // Device buffer for the scoring matrix for all windows.
    ScoreT* scores;
    /// Buffer for storing per row band start location in absolute score matrix for adaptive banding
    SizeT* band_starts;
    /// Buffer for storing per row band widths for adaptive banding
    SizeT* band_widths;
    /// Buffer for storing per row band start location in this score matrix for adaptive banding
    int64_t* band_head_indices;
    /// Buffer for storing max score index per row
    SizeT* band_max_indices;

    // preallocated size of scores buffer
    size_t scorebuf_alloc_size = 0;

    // Device buffers for alignment backtrace
    SizeT* alignment_graph;
    SizeT* alignment_read;
};

template <typename SizeT>
struct GraphDetails
{
    // Device buffer to store nodes of the graph. The node itself is the base
    // (A, T, C, G) and the id of the node is it's position in the buffer.
    uint8_t* nodes;

    // Device buffer to store the list of nodes aligned to a
    // specific node in the graph.
    SizeT* node_alignments;
    uint16_t* node_alignment_count;

    // Device buffer to store incoming edges to a node.
    SizeT* incoming_edges;
    uint16_t* incoming_edge_count;

    // Device buffer to store outgoing edges from a node.
    SizeT* outgoing_edges;
    uint16_t* outgoing_edge_count;

    // Devices buffers to store incoming and outgoing edge weights.
    uint16_t* incoming_edge_weights;
    uint16_t* outgoing_edge_weights;

    // Device buffer to store the topologically sorted graph. Each element
    // of this buffer is an ID of the node.
    SizeT* sorted_poa;

    // Device buffer that maintains a mapping between the node ID and its
    // position in the topologically sorted graph.
    SizeT* sorted_poa_node_map;

    // Device buffer to store distance of each graph node to the head node(s), used in adaptive-banding alignment
    SizeT* node_distance_to_head;

    // Device buffer used during topological sort to store incoming
    // edge counts for nodes.
    uint16_t* sorted_poa_local_edge_count;

    // Device buffer to store scores calculated during traversal
    // of graph for consensus generation.
    int32_t* consensus_scores;

    // Device buffer to store the predecessors of nodes during
    // graph traversal.
    SizeT* consensus_predecessors;

    // Device buffer to store node marks when performing spoa accurate topsort.
    uint8_t* node_marks;

    // Device buffer to store check for aligned nodes.
    bool* check_aligned_nodes;

    // Device buffer to store stack for nodes to be visited.
    SizeT* nodes_to_visit;

    // Device buffer for storing coverage of each node in graph.
    uint16_t* node_coverage_counts;

    uint16_t* outgoing_edges_coverage;
    uint16_t* outgoing_edges_coverage_count;
    SizeT* node_id_to_msa_pos;
};

template <typename SeqT>
struct SeqT4
{
    SeqT r0, r1, r2, r3;
};

template <typename ScoreT>
struct ScoreT4
{
    ScoreT s0, s1, s2, s3;
};

template <>
struct __align__(4) ScoreT4<int16_t>
{
    int16_t s0, s1, s2, s3;
};

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
