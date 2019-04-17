#pragma once

#include <memory>
#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>

#include <cuda_runtime_api.h>

#define CU_CHECK_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU Error:: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

namespace nvidia {

namespace cudapoa {

class WindowDetails;

enum status
{
    CUDAPOA_SUCCESS = 0,
    CUDAPOA_EXCEEDED_MAXIMUM_POAS,
    CUDAPOA_EXCEEDED_MAXIMUM_SEQUENCE_SIZE,
    CUDAPOA_EXCEEDED_MAXIMUM_SEQUENCES_PER_POA,
    UNKNOWN_FAILURE
};

class Batch
{
    const uint32_t NUM_THREADS = 64;

public:
    Batch(uint32_t max_poas, uint32_t max_sequences_per_poa, uint32_t device);
    ~Batch();

    // Add new partial order alignment to batch.
    status add_poa();

    // Add sequence to last partial order alignment.
    status add_seq_to_poa(const char* seq, uint32_t seq_len);

    // Get total number of partial order alignments in batch.
    uint32_t get_total_poas() const;

    // Run partial order alignment algorithm over all POAs.
    void generate_poa();

    // Get the consensus for each POA.
    void get_consensus(std::vector<std::string>& consensus,
            std::vector<std::vector<uint16_t>>& coverage);

    // Set GPU device to run batch on.
    void set_device_id(uint32_t);

    // Set CUDA stream for GPU device.
    void set_cuda_stream(cudaStream_t stream);

    // Return batch ID.
    uint32_t batch_id() const;

    // Reset batch. Must do before re-using batch.
    void reset();

protected:
    // Maximum POAs to process in batch.
    uint32_t max_poas_ = 0;

    // Maximum sequences per POA.
    uint32_t max_sequences_per_poa_ = 0;

    // GPU Device ID
    uint32_t device_id_ = 0;

    // CUDA stream for launching kernels.
    cudaStream_t stream_;

    // Host buffer for storing consensus.
    std::unique_ptr<uint8_t[]> consensus_h_;

    // Device buffer pointer for storing consensus.
    uint8_t *consensus_d_;

    // Host and device buffer pointer for input data.
    uint8_t *inputs_h_;
    uint8_t *inputs_d_;

    // Host buffer pointfer number of sequences per window.
    uint8_t * num_sequences_per_window_h_;

    // Host and device buffer for sequence lengths.
    uint16_t * sequence_lengths_h_;
    uint16_t * sequence_lengths_d_;

    // Host and device buffer pointers that hold Window Details struct.
    WindowDetails * window_details_d_;
    WindowDetails * window_details_h_;

    // Device buffer for the scoring matrix for all windows.
    int16_t* scores_d_;

    // Device buffers for alignment backtrace.
    // i for graph
    // j for sequence
    int16_t* alignment_graph_d_;
    int16_t* alignment_read_d_;

    // Device buffer to store nodes of the graph. The node itself is the base
    // (A, T, C, G) and the id of the node is it's position in the buffer.
    uint8_t* nodes_d_;

    // Device buffer to store the list of nodes aligned to a 
    // specific node in the graph.
    uint16_t* node_alignments_d_;
    uint16_t* node_alignment_count_d_;

    // Device buffer to store incoming edges to a node.
    uint16_t* incoming_edges_d_;
    uint16_t* incoming_edge_count_d_;

    // Device buffer to store outgoing edges from a node.
    uint16_t* outgoing_edges_d_;
    uint16_t* outgoing_edge_count_d_;

    // Devices buffers to store incoming and outgoing edge weights.
    uint16_t* incoming_edges_weights_d_;
    uint16_t* outoing_edges_weights_d_;

    // Device buffer to store the topologically sorted graph. Each element
    // of this buffer is an ID of the node.
    uint16_t* sorted_poa_d_;

    // Device buffer that maintains a mapping between the node ID and its
    // position in the topologically sorted graph.
    uint16_t* sorted_poa_node_map_d_;

    // Device buffer used during topological sort to store incoming
    // edge counts for nodes.
    uint16_t* sorted_poa_local_edge_count_d_;

    // Device buffer to store scores calculated during traversal
    // of graph for consensus generation.
    int32_t* consensus_scores_d_;

    // Device buffer to store the predecessors of nodes during
    // graph traversal.
    int16_t* consensus_predecessors_d_;

    // Device buffer to store node marks when performing spoa accurate topsort.
    uint8_t* node_marks_d_;

    // Device buffer to store check for aligned nodes.
    bool* check_aligned_nodes_d_;

    // Device buffer to store stack for nodes to be visited.
    uint16_t* nodes_to_visit_d_;

    // Buffer for coverage of consensus.
    uint16_t* coverage_h_;
    uint16_t* coverage_d_;

    // Device buffer for storing coverage of each node in graph.
    uint16_t* node_coverage_counts_d_;

    // Static batch count used to generate batch IDs.
    static uint32_t batches;

    // Batch ID.
    uint32_t bid_ = 0;

    // Total POAs added.
    uint32_t poa_count_ = 0;

    // Number of nucleotides already already inserted.
    uint32_t num_nucleotides_copied_ = 0;

    // Global sequence index.
    uint32_t global_sequence_idx_ = 0;
};

}

}
