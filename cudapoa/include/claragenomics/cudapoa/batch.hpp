/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <claragenomics/cudapoa/cudapoa.hpp>

#include <claragenomics/utils/graph.hpp>

#include <memory>
#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>
#include <cuda_runtime_api.h>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claragenomics
{

namespace cudapoa
{

/// \addtogroup cudapoa
/// \{

/// A structure to represent a sequence entry.
struct Entry
{
    /// Pointer to string representing sequence.
    const char* seq;
    /// Pointer to array of weight per base in sequence.
    const int8_t* weights;
    /// Length of sequence.
    int32_t length;
};

/// A type defining the set and order of Entry's in which a POA is processed.
typedef std::vector<Entry> Group;

/// A structure to hold  upper limits for data size processed in POA batches
struct BatchSize
{
    /// Maximum number of elements in a sequence
    int32_t max_sequence_size;
    /// Maximum size of final consensus
    int32_t max_concensus_size;
    /// Maximum number of nodes in a graph, 1 graph per window
    int32_t max_nodes_per_window;
    /// Maximum number of nodes in a graph, 1 graph per window in banded mode
    int32_t max_nodes_per_window_banded;
    /// Maximum vertical dimension of scoring matrix, which stores graph
    int32_t max_matrix_graph_dimension = max_nodes_per_window;
    /// Maximum vertical dimension of scoring matrix, which stores graph
    int32_t max_matrix_graph_dimension_banded = max_nodes_per_window_banded;
    /// Maximum horizontal dimension of scoring matrix, which stores sequences
    int32_t max_matrix_sequence_dimension = max_sequence_size;
    /// Maximum number of equences per POA group
    int32_t max_sequences_per_poa;

    /// constructor- set upper limit parameters based on max_sequence_size
    BatchSize(int32_t max_seq_sz = 1024, int32_t max_seq_per_poa = 100)
        /// ensure a 4-byte boundary alignment for any allocated buffer
        : max_sequence_size(max_seq_sz)
        , max_concensus_size(2 * max_sequence_size)
        , max_nodes_per_window(cudautils::align<int32_t, 4>(3 * max_sequence_size))
        , max_nodes_per_window_banded(cudautils::align<int32_t, 4>(4 * max_sequence_size))
        , max_matrix_graph_dimension(cudautils::align<int32_t, 4>(max_nodes_per_window))
        , max_matrix_graph_dimension_banded(cudautils::align<int32_t, 4>(max_nodes_per_window_banded))
        , max_matrix_sequence_dimension(cudautils::align<int32_t, 4>(max_sequence_size))
        , max_sequences_per_poa(max_seq_per_poa)

    {
        throw_on_negative(max_seq_sz, "max_sequence_size cannot be negative.");
        throw_on_negative(max_seq_per_poa, "max_sequences_per_poa cannot be negative.");
    }

    /// constructor- set all parameters separately
    BatchSize(int32_t max_seq_sz, int32_t max_concensus_sz, int32_t max_nodes_per_w,
              int32_t max_nodes_per_w_banded, int32_t max_seq_per_poa)
        /// ensure a 4-byte boundary alignment for any allocated buffer
        : max_sequence_size(max_seq_sz)
        , max_concensus_size(max_concensus_sz)
        , max_nodes_per_window(cudautils::align<int32_t, 4>(max_nodes_per_w))
        , max_nodes_per_window_banded(cudautils::align<int32_t, 4>(max_nodes_per_w_banded))
        , max_matrix_graph_dimension(cudautils::align<int32_t, 4>(max_nodes_per_window))
        , max_matrix_graph_dimension_banded(cudautils::align<int32_t, 4>(max_nodes_per_window_banded))
        , max_matrix_sequence_dimension(cudautils::align<int32_t, 4>(max_sequence_size))
        , max_sequences_per_poa(max_seq_per_poa)
    {
        throw_on_negative(max_seq_sz, "max_sequence_size cannot be negative.");
        throw_on_negative(max_concensus_sz, "max_concensus_size cannot be negative.");
        throw_on_negative(max_nodes_per_w, "max_nodes_per_window cannot be negative.");
        throw_on_negative(max_nodes_per_w_banded, "max_nodes_per_window_banded cannot be negative.");
        throw_on_negative(max_seq_per_poa, "max_sequences_per_poa cannot be negative.");

        if (max_nodes_per_window < max_sequence_size)
            throw std::invalid_argument("max_nodes_per_window should be greater than or equal to max_sequence_size.");
        if (max_nodes_per_window_banded < max_sequence_size)
            throw std::invalid_argument("max_nodes_per_window should be greater than or equal to max_sequence_size.");
        if (max_concensus_size < max_sequence_size)
            throw std::invalid_argument("max_concensus_size should be greater than or equal to max_sequence_size.");
    }
};

/// \class Batch
/// Batched GPU CUDA POA object
class Batch
{
public:
    /// \brief CudapoaBatch has a custom dtor, so declare ~Batch virtual and give it a default implementation
    virtual ~Batch() = default;

    /// \brief Add a new group to the batch to run POA algorithm on. Based on the constraints
    ///        of the batch, now all entries in a group may be added. This will be reflected in
    ///        the per_seq_status of the call. Those entries that were added will be shown with a success.
    ///
    /// \param per_seq_status Reference to an output vector of StatusType that holds
    ///                       the processing status of each entry in the group.
    ///                       NOTE: This API clears old entries in the vector.
    /// \param poa_group      Vector of Entry's to process in POA. Based on the constraints
    ///                       of the batch, not all entries in a group may be added.
    ///                       This will be reflected in the per_seq_status of the call. Those entries that were
    ///                       added will show a success status. The POA algorithm will run with
    ///                       the sequences that were added.
    ///
    /// \return Status representing whether PoaGroup was successfully added to batch.
    virtual StatusType add_poa_group(std::vector<StatusType>& per_seq_status,
                                     const Group& poa_group) = 0;

    /// \brief Get total number of partial order alignments in batch.
    ///
    /// \return Total POAs in batch.
    virtual int32_t get_total_poas() const = 0;

    /// \brief Run partial order alignment algorithm over all POAs.
    virtual void generate_poa() = 0;

    /// \brief Get the consensus for each POA.
    ///
    /// \param consensus Reference to vector where consensus strings
    ///                  will be returned
    /// \param coverage Reference to vector where coverage of each
    ///                 base in each consensus string is returned
    /// \param output_status Reference to vector where the errors
    ///                 during kernel execution is captured
    ///
    /// \return Status indicating whether consensus generation is available for this batch.
    virtual StatusType get_consensus(std::vector<std::string>& consensus,
                                     std::vector<std::vector<uint16_t>>& coverage,
                                     std::vector<claragenomics::cudapoa::StatusType>& output_status) = 0;

    /// \brief Get the multiple sequence alignments for each POA.
    ///
    /// \param msa Reference to vector where msa strings of each
    ///                 poa is returned
    /// \param output_status Reference to vector where the errors
    ///                 during kernel execution is captured
    ///
    /// \return Status indicating whether MSA generation is available for this batch.
    virtual StatusType get_msa(std::vector<std::vector<std::string>>& msa,
                               std::vector<StatusType>& output_status) = 0;

    /// \brief Get the graph representation for each POA.
    ///
    /// \param graphs Reference to a vector where directed graph of each poa
    ///               is returned.
    /// \param output_status Reference to vector where the errors
    ///                 during kernel execution is captured
    virtual void get_graphs(std::vector<DirectedGraph>& graphs,
                            std::vector<StatusType>& output_status) = 0;

    /// \brief Return batch ID.
    ///
    /// \return Batch ID
    virtual int32_t batch_id() const = 0;

    /// \brief Reset batch. Must do before re-using batch.
    virtual void reset() = 0;
};

/// \brief Creates a new CUDA Batch object.
///
/// \param device_id GPU device on which to run CUDA POA algorithm
/// \param stream CUDA stream to use on GPU
/// \param max_mem Maximum GPU memory to use for this batch.
/// \param output_mask Which outputs to produce from POA (msa, consensus)
/// \param batch_size Defines upper limits for size of a POA batch, i.e. sequence length and other related parameters
/// \param gap_score Score to be assigned to a gap
/// \param mismatch_score Score to be assigned to a mismatch
/// \param match_score Score to be assigned for a match
/// \param cuda_banded_alignment Whether to use banded alignment
///
/// \return Returns a unique pointer to a new Batch object
std::unique_ptr<Batch> create_batch(int32_t device_id,
                                    cudaStream_t stream,
                                    size_t max_mem,
                                    int8_t output_mask,
                                    const BatchSize& batch_size,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score,
                                    bool cuda_banded_alignment);

/// \}

} // namespace cudapoa

} // namespace claragenomics
