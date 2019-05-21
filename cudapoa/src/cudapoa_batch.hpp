#pragma once

#include "cudapoa/batch.hpp"

#include <memory>
#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>

#include <cuda_runtime_api.h>

namespace genomeworks
{

namespace cudapoa
{

class WindowDetails;

class AlignmentDetails;

class GraphDetails;

class OutputDetails;

class InputDetails;

/// \addtogroup cudapoa
/// \{

/// \class
/// Batched GPU CUDA POA object
class CudapoaBatch : public Batch
{
    const uint32_t NUM_THREADS = 64;

public:
    CudapoaBatch(uint32_t max_poas, uint32_t max_sequences_per_poa, uint32_t device_id, int16_t gap_score = -8, int16_t mismatch_score = -6, int16_t match_score = 8, bool cuda_banded_alignment = false);
    ~CudapoaBatch();

    // Add new partial order alignment to batch.
    StatusType add_poa();

    // Add sequence to last partial order alignment.
    StatusType add_seq_to_poa(const char* seq, const uint8_t* weights, uint32_t seq_len);

    // Get total number of partial order alignments in batch.
    uint32_t get_total_poas() const;

    // Run partial order alignment algorithm over all POAs.
    void generate_poa();

    // Get the consensus for each POA.
    void get_consensus(std::vector<std::string>& consensus,
                       std::vector<std::vector<uint16_t>>& coverage,
                       std::vector<genomeworks::cudapoa::StatusType>& output_status);

    // Set CUDA stream for GPU device.
    void set_cuda_stream(cudaStream_t stream);

    // Return batch ID.
    uint32_t batch_id() const;

    // Reset batch. Must do before re-using batch.
    void reset();

protected:
    // Print debug message with batch specific formatting.
    void print_batch_debug_message(const std::string& message);

    // Allocate buffers for output details
    void initialize_output_details();

    // Free buffers for output details
    void free_output_details();

    // Allocate buffers for alignment details
    void initialize_alignment_details(bool banded_alignment);

    // Free buffers for alignment details
    void free_alignment_details();

    // Allocate buffers for graph details
    void initialize_graph_details(bool banded_alignment);

    // Free buffers for graph details
    void free_graph_details();

    // Allocate buffers for input details
    void initialize_input_details();

    // Free buffers for input details
    void free_input_details();

protected:
    // Maximum POAs to process in batch.
    uint32_t max_poas_ = 0;

    // Maximum sequences per POA.
    uint32_t max_sequences_per_poa_ = 0;

    // GPU Device ID
    uint32_t device_id_ = 0;

    // Gap, mismatch and match scores for NW dynamic programming loop.
    int16_t gap_score_;
    int16_t mismatch_score_;
    int16_t match_score_;

    // CUDA stream for launching kernels.
    cudaStream_t stream_;

    // Host and device buffer for output data.
    OutputDetails* output_details_h_;
    OutputDetails* output_details_d_;

    // Host and device buffer pointer for input data.
    InputDetails* input_details_d_;
    InputDetails* input_details_h_;

    // Device buffer struct for alignment details
    AlignmentDetails* alignment_details_d_;

    // Device buffer struct for graph details
    GraphDetails* graph_details_d_;

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

    // Use banded POA alignment
    bool banded_alignment_;
};

/// \}

} // namespace cudapoa

} // namespace genomeworks
