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

#include <claraparabricks/genomeworks/cudapoa/cudapoa.hpp>

#include <claraparabricks/genomeworks/utils/graph.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

#include <memory>
#include <vector>
#include <stdint.h>
#include <string>
#include <iostream>
#include <cuda_runtime_api.h>

namespace claraparabricks
{

namespace genomeworks
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
struct BatchConfig
{
    /// Maximum number of elements in a sequence
    int32_t max_sequence_size;
    /// Maximum size of final consensus
    int32_t max_consensus_size;
    /// Maximum number of nodes in a POA graph, one graph per window
    int32_t max_nodes_per_graph;
    /// Maximum vertical dimension of scoring matrix, which stores POA graph
    int32_t matrix_graph_dimension;
    /// Maximum horizontal dimension of scoring matrix, which stores part of sequences used in scores matrix computation
    int32_t matrix_sequence_dimension;
    /// Band-width used in banded alignment, it also defines minimum band-width in adaptive alignment
    int32_t alignment_band_width;
    /// Maximum number of equences per POA group
    int32_t max_sequences_per_poa;
    /// Banding mode: full, static, adaptive
    BandMode band_mode;

    /// constructor- set upper limit parameters based on max_seq_sz and band_width
    BatchConfig(int32_t max_seq_sz = 1024, int32_t max_seq_per_poa = 100, int32_t band_width = 256, BandMode banding = BandMode::full_band);

    /// constructor- set all parameters separately
    BatchConfig(int32_t max_seq_sz, int32_t max_consensus_sz, int32_t max_nodes_per_w,
                int32_t band_width, int32_t max_seq_per_poa, int32_t matrix_seq_dim, BandMode banding);
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
                                     std::vector<genomeworks::cudapoa::StatusType>& output_status) = 0;

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
/// \param device_id                GPU device on which to run CUDA POA algorithm
/// \param stream                   CUDA stream to use on GPU
/// \param max_gpu_mem              Maximum GPU memory to use for this batch.
/// \param output_mask              which outputs to produce from POA (msa, consensus)
/// \param batch_size               defines upper limits for size of a POA batch, i.e. sequence length and other related parameters
/// \param gap_score                score to be assigned to a gap
/// \param mismatch_score           score to be assigned to a mismatch
/// \param match_score              score to be assigned for a match
///
/// \return Returns a unique pointer to a new Batch object
std::unique_ptr<Batch> create_batch(int32_t device_id,
                                    cudaStream_t stream,
                                    size_t max_gpu_mem,
                                    int8_t output_mask,
                                    const BatchConfig& batch_size,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score);

/// \}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
