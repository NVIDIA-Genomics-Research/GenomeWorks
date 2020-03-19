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
struct UpperLimits
{
    // Maximum number of elements in a sequence
    uint32_t max_sequence_size = 1024;
    // Maximum size of final consensus
    uint32_t max_concensus_size = 1024;
    // Maximum number of nodes in a graph, 1 graph per window
    uint32_t max_nodes_per_window        = 3072;
    uint32_t max_nodes_per_window_banded = 4096;
    // Maximum vertical dimension of scoring matrix, which stores graph
    // Adding 4 elements more to ensure a 4byte boundary alignment for any allocated buffer
    uint32_t max_matrix_graph_dimension        = max_nodes_per_window + 4;
    uint32_t max_matrix_graph_dimension_banded = max_nodes_per_window_banded + 4;
    // Maximum horizontal dimension of scoring matrix, which stores sequences
    // Adding 4 elements more to ensure a 4byte boundary alignment for any allocated buffer
    uint32_t max_matrix_sequence_dimension = max_sequence_size + 4;
    /// ToDO add banded alignment score matrix dimension parameters (maybe?)
    //uint32_t max_matrix_sequence_dimension_banded;

    // set upper limit parameters based on max_sequence_size
    void setLimits(const uint32_t max_seq_sz)
    {
        max_sequence_size           = max_seq_sz;
        max_concensus_size          = max_sequence_size;
        max_nodes_per_window        = 3 * max_sequence_size;
        max_nodes_per_window_banded = 4 * max_sequence_size;
        // Adding 4 elements more to ensure a 4byte boundary alignment for any allocated buffer
        max_matrix_graph_dimension        = max_nodes_per_window + 4;
        max_matrix_graph_dimension_banded = max_nodes_per_window_banded + 4;
        max_matrix_sequence_dimension     = max_sequence_size + 4;
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
/// \param max_sequences_per_poa Maximum number of sequences per POA
/// \param device_id GPU device on which to run CUDA POA algorithm
/// \param stream CUDA stream to use on GPU
/// \param max_mem Maximum GPU memory to use for this batch.
/// \param output_mask Which outputs to produce from POA (msa, consensus)
/// \param max_limits Defines upper limits for size of input data, i.e. sequence length and other related parameters
/// \param gap_score Score to be assigned to a gap
/// \param mismatch_score Score to be assigned to a mismatch
/// \param match_score Score to be assigned for a match
/// \param cuda_banded_alignment Whether to use banded alignment
///
/// \return Returns a unique pointer to a new Batch object
std::unique_ptr<Batch> create_batch(int32_t max_sequences_per_poa,
                                    int32_t device_id,
                                    cudaStream_t stream,
                                    size_t max_mem,
                                    int8_t output_mask,
                                    UpperLimits max_limits,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score,
                                    bool cuda_banded_alignment);

/// \}

} // namespace cudapoa

} // namespace claragenomics
