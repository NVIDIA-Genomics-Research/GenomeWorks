#pragma once

#include <cudapoa/cudapoa.hpp>

#include <memory>
#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>

#include <cuda_runtime_api.h>

namespace genomeworks {

namespace cudapoa {


/// \class
/// Batched GPU CUDA POA object
class Batch
{
    // const uint32_t NUM_THREADS = 64;
    
public:
    //CudapoaBatch has a custom dtor, so declare ~Batch virtual and give it a default implementation
    virtual ~Batch() = default;

    // Add new partial order alignment to batch.
    virtual StatusType add_poa() = 0;

    // Add sequence to last partial order alignment.
    virtual StatusType add_seq_to_poa(const char* seq, uint32_t seq_len) = 0;

    // Get total number of partial order alignments in batch.
    virtual uint32_t get_total_poas() const = 0;

    // Run partial order alignment algorithm over all POAs.
    virtual void generate_poa() = 0;

    // Get the consensus for each POA.
    virtual void get_consensus(std::vector<std::string>& consensus,
            std::vector<std::vector<uint16_t>>& coverage) = 0;

    // Set CUDA stream for GPU device.
    virtual void set_cuda_stream(cudaStream_t stream) = 0;

    // Return batch ID.
    virtual uint32_t batch_id() const = 0;

    // Reset batch. Must do before re-using batch.
    virtual void reset() = 0;

};

// create_batch - return a pointer to Batch object
std::unique_ptr<Batch> create_batch(uint32_t max_poas, uint32_t max_sequences_per_poa, uint32_t device_id, int16_t gap_score = -8, int16_t mismatch_score = -6, int16_t match_score = 8);

/// \}

} // namespace cudapoa

} // namespace genomeworks
