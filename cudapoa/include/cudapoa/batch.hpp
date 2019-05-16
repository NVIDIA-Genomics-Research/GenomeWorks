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


/// \class Batch
/// Batched GPU CUDA POA object
class Batch
{
    // const uint32_t NUM_THREADS = 64;
    
public:
    /// \brief CudapoaBatch has a custom dtor, so declare ~Batch virtual and give it a default implementation
    virtual ~Batch() = default;

    /// \brief Add new partial order alignment to batch.
    virtual StatusType add_poa() = 0;

    /// \brief Add sequence to last partial order alignment.
    ///
    /// \param seq New sequence to be added to most recent POA
    /// \param weights Weight per base in sequence. If no weights are supplied,
    ///                per base weight defaults to 1.
    /// \param seq_len Length of sequence added
    ///
    /// \return Whether sequence could be successfully added to POA
    virtual StatusType add_seq_to_poa(const char* seq, const uint8_t* weights, uint32_t seq_len) = 0;

    /// \brief Get total number of partial order alignments in batch.
    ///
    /// \return Total POAs in batch.
    virtual uint32_t get_total_poas() const = 0;

    /// \brief Run partial order alignment algorithm over all POAs.
    virtual void generate_poa() = 0;

    /// \brief Get the consensus for each POA.
    ///
    /// \param consensus Reference to vector where consensus strings
    ///                  will be returned
    /// \param coverage Reference to vector where coverage of each
    ///                 base in each consensus string is returned
    virtual void get_consensus(std::vector<std::string>& consensus,
            std::vector<std::vector<uint16_t>>& coverage) = 0;

    /// \brief Set CUDA stream for GPU device.
    virtual void set_cuda_stream(cudaStream_t stream) = 0;

    /// \brief Return batch ID.
    ///
    /// \return Batch ID
    virtual uint32_t batch_id() const = 0;

    /// \brief Reset batch. Must do before re-using batch.
    virtual void reset() = 0;

};

/// \brief Creates a new CUDA Batch object.
///
/// \param max_poas Maximum number of POAs that can be added to the batch
/// \param max_sequences_per_poa Maximum number of sequences per POA
/// \param device_id GPU device on which to run CUDA POA algorithm
/// \param gap_score Score to be assigned to a gap
/// \param mismatch_score Score to be assigned to a mismatch
/// \param match_score Score to be assigned for a match
/// \param cuda_banded_alignment Whether to use banded alignment
///
/// \return Returns a unique pointer to a new Batch object
std::unique_ptr<Batch> create_batch(uint32_t max_poas, uint32_t max_sequences_per_poa, uint32_t device_id, int16_t gap_score = -8, int16_t mismatch_score = -6, int16_t match_score = 8, bool cuda_banded_alignment=false);

/// \}

} // namespace cudapoa

} // namespace genomeworks
