/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#define GW_LOG_LEVEL gw_log_level_info

#include "allocate_block.hpp"

#include <cudautils/cudautils.hpp>
#include <logging/logging.hpp>
#include <utils/signed_integer_utils.hpp>

namespace cga
{

namespace cudapoa
{

BatchBlock::BatchBlock(int32_t device_id, int32_t max_poas, int32_t max_sequences_per_poa, int8_t output_mask, bool banded_alignment)
    : max_poas_(throw_on_negative(max_poas, "Maximum POAs in block has to be non-negative"))
    , max_sequences_per_poa_(throw_on_negative(max_sequences_per_poa, "Maximum sequences per POA has to be non-negative"))
    , banded_alignment_(banded_alignment)
    , device_id_(throw_on_negative(device_id, "Device ID has to be non-negative"))
    , output_mask_(output_mask)
{
    output_size_               = max_poas_ * CUDAPOA_MAX_CONSENSUS_SIZE;
    input_size_                = max_poas_ * max_sequences_per_poa_ * CUDAPOA_MAX_SEQUENCE_SIZE;
    matrix_sequence_dimension_ = banded_alignment_ ? CUDAPOA_BANDED_MAX_MATRIX_SEQUENCE_DIMENSION : CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION;
    max_graph_dimension_       = banded_alignment_ ? CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION_BANDED : CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION;
    max_nodes_per_window_      = banded_alignment_ ? CUDAPOA_MAX_NODES_PER_WINDOW_BANDED : CUDAPOA_MAX_NODES_PER_WINDOW;

    // Set CUDA device
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));

    // calculate size
    calculate_size();

    // allocation
    GW_CU_CHECK_ERR(cudaHostAlloc((void**)&block_data_h_, total_h_, cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaMalloc((void**)&block_data_d_, total_d_));
}

BatchBlock::~BatchBlock()
{
    GW_CU_CHECK_ERR(cudaFree(block_data_d_));
    GW_CU_CHECK_ERR(cudaFreeHost(block_data_h_));
}

uint8_t* BatchBlock::get_block_host()
{
    return block_data_h_;
}

uint8_t* BatchBlock::get_block_device()
{
    return block_data_d_;
}

void BatchBlock::calculate_size()
{
    // for output - host
    total_h_ += sizeof(OutputDetails);                                                                          // output_details_h_
    total_h_ += output_size_ * sizeof(uint8_t);                                                                 // output_details_h_->consensus
    total_h_ += (output_mask_ & OutputType::consensus) ? output_size_ * sizeof(uint16_t) : 0;                   // output_details_h_->coverage
    total_h_ += (output_mask_ & OutputType::msa) ? output_size_ * max_sequences_per_poa_ * sizeof(uint8_t) : 0; // output_details_h_->multiple_sequence_alignments
    total_h_ += sizeof(OutputDetails);                                                                          // output_details_d_
    // for output - device
    total_d_ += output_size_ * sizeof(uint8_t);                                                                 // output_details_d_->consensus
    total_d_ += (output_mask_ & OutputType::consensus) ? output_size_ * sizeof(uint16_t) : 0;                   // output_details_d_->coverage
    total_d_ += (output_mask_ & OutputType::msa) ? output_size_ * max_sequences_per_poa_ * sizeof(uint8_t) : 0; // output_details_d_->multiple_sequence_alignments

    // for input - host
    total_h_ += sizeof(InputDetails);                                                                         // input_details_h_
    total_h_ += input_size_ * sizeof(uint8_t);                                                                // input_details_h_->sequences
    total_h_ += input_size_ * sizeof(int8_t);                                                                 // input_details_h_->base_weights
    total_h_ += max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t);                                        // input_details_h_->sequence_lengths
    total_h_ += max_poas_ * sizeof(WindowDetails);                                                            // input_details_h_->window_details
    total_h_ += (output_mask_ & OutputType::msa) ? max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t) : 0; // input_details_h_->sequence_begin_nodes_ids

    total_h_ += sizeof(InputDetails); // input_details_d_
    // for input - device
    total_d_ += input_size_ * sizeof(uint8_t);                                                                // input_details_d_->sequences
    total_d_ += input_size_ * sizeof(int8_t);                                                                 // input_details_d_->base_weights
    total_d_ += max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t);                                        // input_details_d_->sequence_lengths
    total_d_ += max_poas_ * sizeof(WindowDetails);                                                            // input_details_d_->window_details
    total_d_ += (output_mask_ & OutputType::msa) ? max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t) : 0; // input_details_d_->sequence_begin_nodes_ids

    // for alignment - host
    total_h_ += sizeof(AlignmentDetails); // alignment_details_d_
    // for alignment - device
    total_d_ += sizeof(int16_t) * max_graph_dimension_ * matrix_sequence_dimension_ * max_poas_; // alignment_details_d_->scores
    total_d_ += sizeof(int16_t) * max_graph_dimension_ * max_poas_;                              // alignment_details_d_->alignment_graph
    total_d_ += sizeof(int16_t) * max_graph_dimension_ * max_poas_;                              // alignment_details_d_->alignment_read

    // for graph - host
    total_h_ += sizeof(GraphDetails); // graph_details_d_
    // for graph - device
    total_d_ += sizeof(uint8_t) * max_nodes_per_window_ * max_poas_;                                                                                           // graph_details_d_->nodes
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_ALIGNMENTS * max_poas_;                                                            // graph_details_d_->node_alignments
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;                                                                                          // graph_details_d_->node_alignment_count
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;                                                                 // graph_details_d_->incoming_edges
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;                                                                                          // graph_details_d_->incoming_edge_count
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;                                                                 // graph_details_d_->outgoing_edges
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;                                                                                          // graph_details_d_->outgoing_edge_count
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;                                                                 // graph_details_d_->incoming_edge_weights
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;                                                                 // graph_details_d_->outgoing_edge_weights
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;                                                                                          // graph_details_d_->sorted_poa
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;                                                                                          // graph_details_d_->sorted_poa_node_map
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;                                                                                          // graph_details_d_->sorted_poa_local_edge_count
    total_d_ += (output_mask_ & OutputType::consensus) ? sizeof(int32_t) * max_nodes_per_window_ * max_poas_ : 0;                                              // graph_details_d_->consensus_scores
    total_d_ += (output_mask_ & OutputType::consensus) ? sizeof(int16_t) * max_nodes_per_window_ * max_poas_ : 0;                                              // graph_details_d_->consensus_predecessors
    total_d_ += sizeof(int8_t) * max_nodes_per_window_ * max_poas_;                                                                                            // graph_details_d_->node_marks
    total_d_ += sizeof(bool) * max_nodes_per_window_ * max_poas_;                                                                                              // graph_details_d_->check_aligned_nodes
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;                                                                                          // graph_details_d_->nodes_to_visit
    total_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;                                                                                          // graph_details_d_->node_coverage_counts
    total_d_ += (output_mask_ & OutputType::msa) ? sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa_ * max_poas_ : 0; // graph_details_d_->outgoing_edges_coverage
    total_d_ += (output_mask_ & OutputType::msa) ? sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_ : 0;                          // graph_details_d_->outgoing_edges_coverage_count
    total_d_ += (output_mask_ & OutputType::msa) ? sizeof(int16_t) * max_nodes_per_window_ * max_poas_ : 0;                                                    // graph_details_d_->node_id_to_msa_pos
}

void BatchBlock::get_output_details(OutputDetails** output_details_h_p, OutputDetails** output_details_d_p)
{
    OutputDetails* output_details_h{};
    OutputDetails* output_details_d{};

    // on host
    output_details_h = reinterpret_cast<OutputDetails*>(&block_data_h_[offset_h_]);
    offset_h_ += sizeof(OutputDetails);
    output_details_h->consensus = &block_data_h_[offset_h_];
    offset_h_ += output_size_ * sizeof(uint8_t);
    if (output_mask_ & OutputType::consensus)
    {
        output_details_h->coverage = reinterpret_cast<uint16_t*>(&block_data_h_[offset_h_]);
        offset_h_ += output_size_ * sizeof(uint16_t);
    }
    if (output_mask_ & OutputType::msa)
    {
        output_details_h->multiple_sequence_alignments = reinterpret_cast<uint8_t*>(&block_data_h_[offset_h_]);
        offset_h_ += output_size_ * max_sequences_per_poa_ * sizeof(uint8_t);
    }

    output_details_d = reinterpret_cast<OutputDetails*>(&block_data_h_[offset_h_]);
    offset_h_ += sizeof(OutputDetails);

    // on device
    output_details_d->consensus = &block_data_d_[offset_d_];
    offset_d_ += output_size_ * sizeof(int8_t);
    if (output_mask_ & OutputType::consensus)
    {
        output_details_d->coverage = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += output_size_ * sizeof(int16_t);
    }
    if (output_mask_ & OutputType::msa)
    {
        output_details_d->multiple_sequence_alignments = reinterpret_cast<uint8_t*>(&block_data_d_[offset_d_]);
        offset_d_ += output_size_ * max_sequences_per_poa_ * sizeof(uint8_t);
    }

    *output_details_h_p = output_details_h;
    *output_details_d_p = output_details_d;
}

void BatchBlock::get_input_details(InputDetails** input_details_h_p, InputDetails** input_details_d_p)
{
    // on host
    InputDetails* input_details_h{};
    InputDetails* input_details_d{};

    input_details_h = reinterpret_cast<InputDetails*>(&block_data_h_[offset_h_]);
    offset_h_ += sizeof(InputDetails);
    input_details_h->sequences = &block_data_h_[offset_h_];
    offset_h_ += input_size_ * sizeof(uint8_t);
    input_details_h->base_weights = reinterpret_cast<int8_t*>(&block_data_h_[offset_h_]);
    offset_h_ += input_size_ * sizeof(int8_t);
    input_details_h->sequence_lengths = reinterpret_cast<uint16_t*>(&block_data_h_[offset_h_]);
    offset_h_ += max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t);
    input_details_h->window_details = reinterpret_cast<WindowDetails*>(&block_data_h_[offset_h_]);
    offset_h_ += max_poas_ * sizeof(WindowDetails);
    if (output_mask_ & OutputType::msa)
    {
        input_details_h->sequence_begin_nodes_ids = reinterpret_cast<uint16_t*>(&block_data_h_[offset_h_]);
        offset_h_ += max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t);
    }

    input_details_d = reinterpret_cast<InputDetails*>(&block_data_h_[offset_h_]);
    offset_h_ += sizeof(InputDetails);

    // on device
    input_details_d->sequences = &block_data_d_[offset_d_];
    offset_d_ += input_size_ * sizeof(uint8_t);
    input_details_d->base_weights = reinterpret_cast<int8_t*>(&block_data_d_[offset_d_]);
    offset_d_ += input_size_ * sizeof(int8_t);
    input_details_d->sequence_lengths = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t);
    input_details_d->window_details = reinterpret_cast<WindowDetails*>(&block_data_d_[offset_d_]);
    offset_d_ += max_poas_ * sizeof(WindowDetails);
    if (output_mask_ & OutputType::msa)
    {
        input_details_d->sequence_begin_nodes_ids = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t);
    }

    *input_details_h_p = input_details_h;
    *input_details_d_p = input_details_d;
}

void BatchBlock::get_alignment_details(AlignmentDetails** alignment_details_d_p)
{
    AlignmentDetails* alignment_details_d{};

    // on host
    alignment_details_d = reinterpret_cast<AlignmentDetails*>(&block_data_h_[offset_h_]);
    offset_h_ += sizeof(AlignmentDetails);

    // on device
    alignment_details_d->scores = reinterpret_cast<int16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(int16_t) * max_graph_dimension_ * matrix_sequence_dimension_ * max_poas_;
    alignment_details_d->alignment_graph = reinterpret_cast<int16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(int16_t) * max_graph_dimension_ * max_poas_;
    alignment_details_d->alignment_read = reinterpret_cast<int16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(int16_t) * max_graph_dimension_ * max_poas_;

    *alignment_details_d_p = alignment_details_d;
}

void BatchBlock::get_graph_details(GraphDetails** graph_details_d_p)
{
    GraphDetails* graph_details_d{};

    // on host
    graph_details_d = reinterpret_cast<GraphDetails*>(&block_data_h_[offset_h_]);
    offset_h_ += sizeof(GraphDetails);

    // on device
    graph_details_d->nodes = &block_data_d_[offset_d_];
    offset_d_ += sizeof(uint8_t) * max_nodes_per_window_ * max_poas_;
    graph_details_d->node_alignments = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_ALIGNMENTS * max_poas_;
    graph_details_d->node_alignment_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    graph_details_d->incoming_edges = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
    graph_details_d->incoming_edge_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    graph_details_d->outgoing_edges = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
    graph_details_d->outgoing_edge_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    graph_details_d->incoming_edge_weights = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
    graph_details_d->outgoing_edge_weights = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
    graph_details_d->sorted_poa = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    graph_details_d->sorted_poa_node_map = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    graph_details_d->sorted_poa_local_edge_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    if (output_mask_ & OutputType::consensus)
    {
        graph_details_d->consensus_scores = reinterpret_cast<int32_t*>(&block_data_d_[offset_d_]);
        offset_d_ += sizeof(int32_t) * max_nodes_per_window_ * max_poas_;
        graph_details_d->consensus_predecessors = reinterpret_cast<int16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += sizeof(int16_t) * max_nodes_per_window_ * max_poas_;
    }

    graph_details_d->node_marks = reinterpret_cast<uint8_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(int8_t) * max_nodes_per_window_ * max_poas_;
    graph_details_d->check_aligned_nodes = reinterpret_cast<bool*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(bool) * max_nodes_per_window_ * max_poas_;
    graph_details_d->nodes_to_visit = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    graph_details_d->node_coverage_counts = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
    offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    if (output_mask_ & OutputType::msa)
    {
        graph_details_d->outgoing_edges_coverage = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa_ * max_poas_;
        graph_details_d->outgoing_edges_coverage_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
        graph_details_d->node_id_to_msa_pos = reinterpret_cast<int16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
    }

    *graph_details_d_p = graph_details_d;
}

} // namespace cudapoa

} // namespace cga
