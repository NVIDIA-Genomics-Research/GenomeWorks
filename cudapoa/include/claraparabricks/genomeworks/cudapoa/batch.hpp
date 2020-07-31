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
    BatchConfig(int32_t max_seq_sz = 1024, int32_t max_seq_per_poa = 100, int32_t band_width = 256, BandMode banding = BandMode::full_band)
        /// ensure a 4-byte boundary alignment for any allocated buffer
        : max_sequence_size(max_seq_sz)
        , max_consensus_size(2 * max_sequence_size)
        /// ensure 128-alignment for band_width size, 128 = CUDAPOA_MIN_BAND_WIDTH
        , alignment_band_width(cudautils::align<int32_t, 128>(band_width))
        , max_sequences_per_poa(max_seq_per_poa)
        , band_mode(banding)
    {
        if (banding == BandMode::full_band)
        {
            max_nodes_per_graph       = cudautils::align<int32_t, 4>(3 * max_sequence_size);
            matrix_graph_dimension    = cudautils::align<int32_t, 4>(max_nodes_per_graph);
            matrix_sequence_dimension = cudautils::align<int32_t, 4>(max_sequence_size);
        }
        else if (banding == BandMode::static_band)
        {
            max_nodes_per_graph    = cudautils::align<int32_t, 4>(4 * max_sequence_size);
            matrix_graph_dimension = cudautils::align<int32_t, 4>(max_nodes_per_graph);
            // 8 = CUDAPOA_BANDED_MATRIX_RIGHT_PADDING
            matrix_sequence_dimension = cudautils::align<int32_t, 4>(alignment_band_width + 8);
        }
        else // BandMode::adaptive_band
        {
            max_nodes_per_graph    = cudautils::align<int32_t, 4>(4 * max_sequence_size);
            matrix_graph_dimension = cudautils::align<int32_t, 4>(2 * max_nodes_per_graph);
            // 8 = CUDAPOA_BANDED_MATRIX_RIGHT_PADDING, *2 is to reserve extra memory for cases with extended band-width
            matrix_sequence_dimension = cudautils::align<int32_t, 4>(2 * (alignment_band_width + 8));
        }

        throw_on_negative(max_seq_sz, "max_sequence_size cannot be negative.");
        throw_on_negative(max_seq_per_poa, "max_sequences_per_poa cannot be negative.");
        throw_on_negative(band_width, "alignment_band_width cannot be negative.");
        if (alignment_band_width != band_width)
        {
            std::cerr << "Band-width should be multiple of 128. The input was changed from " << band_width << " to " << alignment_band_width << std::endl;
        }
    }

    /// constructor- set all parameters separately
    BatchConfig(int32_t max_seq_sz, int32_t max_consensus_sz, int32_t max_nodes_per_w,
                int32_t band_width, int32_t max_seq_per_poa, BandMode banding)
        /// ensure a 4-byte boundary alignment for any allocated buffer
        : max_sequence_size(max_seq_sz)
        , max_consensus_size(max_consensus_sz)
        , max_nodes_per_graph(cudautils::align<int32_t, 4>(max_nodes_per_w))
        , matrix_graph_dimension(cudautils::align<int32_t, 4>(max_nodes_per_graph))
        , matrix_sequence_dimension(cudautils::align<int32_t, 4>(max_sequence_size))
        /// ensure 128-alignment for band_width size, 128 = CUDAPOA_MIN_BAND_WIDTH
        , alignment_band_width(cudautils::align<int32_t, 128>(band_width))
        , max_sequences_per_poa(max_seq_per_poa)
        , band_mode(banding)
    {
        throw_on_negative(max_seq_sz, "max_sequence_size cannot be negative.");
        throw_on_negative(max_consensus_sz, "max_consensus_size cannot be negative.");
        throw_on_negative(max_nodes_per_w, "max_nodes_per_graph cannot be negative.");
        throw_on_negative(max_seq_per_poa, "max_sequences_per_poa cannot be negative.");
        throw_on_negative(band_width, "alignment_band_width cannot be negative.");

        if (max_nodes_per_graph < max_sequence_size)
            throw std::invalid_argument("max_nodes_per_graph should be greater than or equal to max_sequence_size.");
        if (max_consensus_size < max_sequence_size)
            throw std::invalid_argument("max_consensus_size should be greater than or equal to max_sequence_size.");
        if (max_sequence_size < alignment_band_width)
            throw std::invalid_argument("alignment_band_width should not be greater than max_sequence_size.");
        if (alignment_band_width != band_width)
        {
            std::cerr << "Band-width should be multiple of 128. The input was changed from " << band_width << " to " << alignment_band_width << std::endl;
        }
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
/// \param max_mem                  Maximum GPU memory to use for this batch.
/// \param output_mask              which outputs to produce from POA (msa, consensus)
/// \param batch_size               defines upper limits for size of a POA batch, i.e. sequence length and other related parameters
/// \param gap_score                score to be assigned to a gap
/// \param mismatch_score           score to be assigned to a mismatch
/// \param match_score              score to be assigned for a match
/// \param banded_alignment         whether to use banded alignment
/// \param adaptive_banded          flag to enable adaptive banded alignment
///
/// \return Returns a unique pointer to a new Batch object
std::unique_ptr<Batch> create_batch(int32_t device_id,
                                    cudaStream_t stream,
                                    size_t max_mem,
                                    int8_t output_mask,
                                    const BatchConfig& batch_size,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score,
                                    bool banded_alignment,
                                    bool adaptive_banded);

/// \}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
