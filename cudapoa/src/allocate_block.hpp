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

#include "cudapoa_structs.cuh"
#include "cudapoa_kernels.cuh"
#include "cudapoa_limits.hpp"

#include <memory>
#include <vector>
#include <stdint.h>
#include <string>
#include <cuda_runtime_api.h>
#include <claraparabricks/genomeworks/cudapoa/batch.hpp>

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/logging/logging.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

#ifndef GW_LOG_LEVEL
#ifndef NDEBUG
/// \brief Defines the logging level used in the current module
#define GW_LOG_LEVEL gw_log_level_debug
#else // NDEBUG
/// \brief Defines the logging level used in the current module
#define GW_LOG_LEVEL gw_log_level_error
#endif // NDEBUG
#endif // GW_LOG_LEVEL

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

template <typename ScoreT, typename SizeT>
class BatchBlock
{
public:
    BatchBlock(int32_t device_id, size_t avail_mem, int8_t output_mask, const BatchConfig& batch_size)
        : max_sequences_per_poa_(throw_on_negative(batch_size.max_sequences_per_poa, "Maximum sequences per POA has to be non-negative"))
        , device_id_(throw_on_negative(device_id, "Device ID has to be non-negative"))
        , output_mask_(output_mask)
    {
        scoped_device_switch dev(device_id_);
        max_nodes_per_window_ = batch_size.max_nodes_per_graph;

        // calculate static and dynamic sizes of buffers needed per POA entry.
        int64_t host_size_fixed, device_size_fixed;
        int64_t host_size_per_poa, device_size_per_poa;
        std::tie(host_size_fixed, device_size_fixed, host_size_per_poa, device_size_per_poa) = calculate_space_per_poa(batch_size);

        // Check minimum requirement for device memory
        size_t minimum_device_mem = device_size_fixed + device_size_per_poa;
        if (avail_mem < minimum_device_mem)
        {
            std::string msg = std::string("Require at least ")
                                  .append(std::to_string(minimum_device_mem))
                                  .append(" bytes of device memory per CUDAPOA batch to process correctly.");
            throw std::runtime_error(msg);
        }

        // Calculate max POAs possible based on available memory.
        int64_t device_size_per_score_matrix = static_cast<int64_t>(batch_size.matrix_sequence_dimension) *
                                               static_cast<int64_t>(batch_size.matrix_graph_dimension) * sizeof(ScoreT);
        max_poas_ = avail_mem / (device_size_per_poa + device_size_per_score_matrix);

        // Update final sizes for block based on calculated maximum POAs.
        output_size_ = max_poas_ * static_cast<int64_t>(batch_size.max_consensus_size);
        input_size_  = max_poas_ * max_sequences_per_poa_ * static_cast<int64_t>(batch_size.max_sequence_size);
        total_h_     = max_poas_ * host_size_per_poa + host_size_fixed;
        total_d_     = avail_mem;

        // Allocate.
        GW_CU_CHECK_ERR(cudaHostAlloc((void**)&block_data_h_, total_h_, cudaHostAllocDefault));
        GW_CU_CHECK_ERR(cudaMalloc((void**)&block_data_d_, total_d_));
    }

    ~BatchBlock()
    {
        GW_CU_CHECK_ERR(cudaFree(block_data_d_));
        GW_CU_CHECK_ERR(cudaFreeHost(block_data_h_));
    }

    void get_output_details(OutputDetails** output_details_h_p, OutputDetails** output_details_d_p)
    {
        OutputDetails* output_details_h{};
        OutputDetails* output_details_d{};

        // on host
        output_details_h = reinterpret_cast<OutputDetails*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(OutputDetails);
        output_details_h->consensus = &block_data_h_[offset_h_];
        offset_h_ += output_size_ * sizeof(*output_details_h->consensus);
        if (output_mask_ & OutputType::consensus)
        {
            output_details_h->coverage = reinterpret_cast<decltype(output_details_h->coverage)>(&block_data_h_[offset_h_]);
            offset_h_ += output_size_ * sizeof(*output_details_h->coverage);
        }
        if (output_mask_ & OutputType::msa)
        {
            output_details_h->multiple_sequence_alignments = reinterpret_cast<decltype(output_details_h->multiple_sequence_alignments)>(&block_data_h_[offset_h_]);
            offset_h_ += output_size_ * max_sequences_per_poa_ * sizeof(*output_details_h->multiple_sequence_alignments);
        }

        output_details_d = reinterpret_cast<OutputDetails*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(OutputDetails);

        // on device
        output_details_d->consensus = &block_data_d_[offset_d_];
        offset_d_ += cudautils::align<int64_t, 8>(output_size_ * sizeof(*output_details_d->consensus));
        if (output_mask_ & OutputType::consensus)
        {
            output_details_d->coverage = reinterpret_cast<decltype(output_details_d->coverage)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(output_size_ * sizeof(*output_details_d->coverage));
        }
        if (output_mask_ & OutputType::msa)
        {
            output_details_d->multiple_sequence_alignments = reinterpret_cast<decltype(output_details_d->multiple_sequence_alignments)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(output_size_ * max_sequences_per_poa_ * sizeof(*output_details_d->multiple_sequence_alignments));
        }

        *output_details_h_p = output_details_h;
        *output_details_d_p = output_details_d;
    }

    void get_input_details(InputDetails<SizeT>** input_details_h_p, InputDetails<SizeT>** input_details_d_p)
    {
        // on host
        InputDetails<SizeT>* input_details_h{};
        InputDetails<SizeT>* input_details_d{};

        input_details_h = reinterpret_cast<InputDetails<SizeT>*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(InputDetails<SizeT>);
        input_details_h->sequences = &block_data_h_[offset_h_];
        offset_h_ += input_size_ * sizeof(*input_details_h->sequences);
        input_details_h->base_weights = reinterpret_cast<decltype(input_details_h->base_weights)>(&block_data_h_[offset_h_]);
        offset_h_ += input_size_ * sizeof(*input_details_h->base_weights);
        input_details_h->sequence_lengths = reinterpret_cast<decltype(input_details_h->sequence_lengths)>(&block_data_h_[offset_h_]);
        offset_h_ += max_poas_ * max_sequences_per_poa_ * sizeof(*input_details_h->sequence_lengths);
        input_details_h->window_details = reinterpret_cast<decltype(input_details_h->window_details)>(&block_data_h_[offset_h_]);
        offset_h_ += max_poas_ * sizeof(*input_details_h->window_details);
        if (output_mask_ & OutputType::msa)
        {
            input_details_h->sequence_begin_nodes_ids = reinterpret_cast<decltype(input_details_h->sequence_begin_nodes_ids)>(&block_data_h_[offset_h_]);
            offset_h_ += max_poas_ * max_sequences_per_poa_ * sizeof(*input_details_h->sequence_begin_nodes_ids);
        }

        input_details_d = reinterpret_cast<InputDetails<SizeT>*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(InputDetails<SizeT>);

        // on device
        input_details_d->sequences = &block_data_d_[offset_d_];
        offset_d_ += cudautils::align<int64_t, 8>(input_size_ * sizeof(*input_details_d->sequences));
        input_details_d->base_weights = reinterpret_cast<decltype(input_details_d->base_weights)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(input_size_ * sizeof(*input_details_d->base_weights));
        input_details_d->sequence_lengths = reinterpret_cast<decltype(input_details_d->sequence_lengths)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(max_poas_ * max_sequences_per_poa_ * sizeof(*input_details_d->sequence_lengths));
        input_details_d->window_details = reinterpret_cast<decltype(input_details_d->window_details)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(max_poas_ * sizeof(*input_details_d->window_details));
        if (output_mask_ & OutputType::msa)
        {
            input_details_d->sequence_begin_nodes_ids = reinterpret_cast<decltype(input_details_d->sequence_begin_nodes_ids)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(max_poas_ * max_sequences_per_poa_ * sizeof(*input_details_d->sequence_begin_nodes_ids));
        }

        *input_details_h_p = input_details_h;
        *input_details_d_p = input_details_d;
    }

    void get_alignment_details(AlignmentDetails<ScoreT, SizeT>** alignment_details_d_p)
    {
        AlignmentDetails<ScoreT, SizeT>* alignment_details_d{};

        // on host
        alignment_details_d = reinterpret_cast<AlignmentDetails<ScoreT, SizeT>*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(AlignmentDetails<ScoreT, SizeT>);

        // on device;
        alignment_details_d->alignment_graph = reinterpret_cast<decltype(alignment_details_d->alignment_graph)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*alignment_details_d->alignment_graph) * max_nodes_per_window_ * max_poas_);
        alignment_details_d->alignment_read = reinterpret_cast<decltype(alignment_details_d->alignment_read)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*alignment_details_d->alignment_read) * max_nodes_per_window_ * max_poas_);
        if (variable_bands_)
        {
            alignment_details_d->band_starts = reinterpret_cast<decltype(alignment_details_d->band_starts)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*alignment_details_d->band_starts) * max_nodes_per_window_ * max_poas_);
            alignment_details_d->band_widths = reinterpret_cast<decltype(alignment_details_d->band_widths)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*alignment_details_d->band_widths) * max_nodes_per_window_ * max_poas_);
            alignment_details_d->band_head_indices = reinterpret_cast<decltype(alignment_details_d->band_head_indices)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*alignment_details_d->band_head_indices) * max_nodes_per_window_ * max_poas_);
            alignment_details_d->band_max_indices = reinterpret_cast<decltype(alignment_details_d->band_max_indices)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*alignment_details_d->band_max_indices) * max_nodes_per_window_ * max_poas_);
        }

        // rest of the available memory is assigned to scores buffer
        alignment_details_d->scorebuf_alloc_size = total_d_ - offset_d_;
        alignment_details_d->scores              = reinterpret_cast<decltype(alignment_details_d->scores)>(&block_data_d_[offset_d_]);
        *alignment_details_d_p                   = alignment_details_d;
    }

    void get_graph_details(GraphDetails<SizeT>** graph_details_d_p, GraphDetails<SizeT>** graph_details_h_p)
    {
        GraphDetails<SizeT>* graph_details_d{};
        GraphDetails<SizeT>* graph_details_h{};

        // on host
        graph_details_h = reinterpret_cast<GraphDetails<SizeT>*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(GraphDetails<SizeT>);
        graph_details_h->nodes = &block_data_h_[offset_h_];
        offset_h_ += sizeof(*graph_details_h->nodes) * max_nodes_per_window_ * max_poas_;
        graph_details_h->incoming_edges = reinterpret_cast<decltype(graph_details_h->incoming_edges)>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(*graph_details_h->incoming_edges) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
        graph_details_h->incoming_edge_weights = reinterpret_cast<decltype(graph_details_h->incoming_edge_weights)>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(*graph_details_h->incoming_edge_weights) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
        graph_details_h->incoming_edge_count = reinterpret_cast<decltype(graph_details_h->incoming_edge_count)>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(*graph_details_h->incoming_edge_count) * max_nodes_per_window_ * max_poas_;
        graph_details_d = reinterpret_cast<GraphDetails<SizeT>*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(GraphDetails<SizeT>);
        graph_details_d->nodes = &block_data_h_[offset_h_];

        // on device
        graph_details_d->nodes = &block_data_d_[offset_d_];
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->nodes) * max_nodes_per_window_ * max_poas_);
        graph_details_d->node_alignments = reinterpret_cast<decltype(graph_details_d->node_alignments)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->node_alignments) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_ALIGNMENTS * max_poas_);
        graph_details_d->node_alignment_count = reinterpret_cast<decltype(graph_details_d->node_alignment_count)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->node_alignment_count) * max_nodes_per_window_ * max_poas_);
        graph_details_d->incoming_edges = reinterpret_cast<decltype(graph_details_d->incoming_edges)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->incoming_edges) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
        graph_details_d->incoming_edge_count = reinterpret_cast<decltype(graph_details_d->incoming_edge_count)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->incoming_edge_count) * max_nodes_per_window_ * max_poas_);
        graph_details_d->outgoing_edges = reinterpret_cast<decltype(graph_details_d->outgoing_edges)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->outgoing_edges) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
        graph_details_d->outgoing_edge_count = reinterpret_cast<decltype(graph_details_d->outgoing_edge_count)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->outgoing_edge_count) * max_nodes_per_window_ * max_poas_);
        graph_details_d->incoming_edge_weights = reinterpret_cast<decltype(graph_details_d->incoming_edge_weights)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->incoming_edge_weights) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
        graph_details_d->outgoing_edge_weights = reinterpret_cast<decltype(graph_details_d->outgoing_edge_weights)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->outgoing_edge_weights) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
        graph_details_d->sorted_poa = reinterpret_cast<decltype(graph_details_d->sorted_poa)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->sorted_poa) * max_nodes_per_window_ * max_poas_);
        graph_details_d->sorted_poa_node_map = reinterpret_cast<decltype(graph_details_d->sorted_poa_node_map)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->sorted_poa_node_map) * max_nodes_per_window_ * max_poas_);
        if (variable_bands_)
        {
            graph_details_d->node_distance_to_head = reinterpret_cast<decltype(graph_details_d->node_distance_to_head)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->node_distance_to_head) * max_nodes_per_window_ * max_poas_);
        }
        graph_details_d->sorted_poa_local_edge_count = reinterpret_cast<decltype(graph_details_d->sorted_poa_local_edge_count)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->sorted_poa_local_edge_count) * max_nodes_per_window_ * max_poas_);
        if (output_mask_ & OutputType::consensus)
        {
            graph_details_d->consensus_scores = reinterpret_cast<decltype(graph_details_d->consensus_scores)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->consensus_scores) * max_nodes_per_window_ * max_poas_);
            graph_details_d->consensus_predecessors = reinterpret_cast<decltype(graph_details_d->consensus_predecessors)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->consensus_predecessors) * max_nodes_per_window_ * max_poas_);
        }

        graph_details_d->node_marks = reinterpret_cast<decltype(graph_details_d->node_marks)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->node_marks) * max_nodes_per_window_ * max_poas_);
        graph_details_d->check_aligned_nodes = reinterpret_cast<decltype(graph_details_d->check_aligned_nodes)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->check_aligned_nodes) * max_nodes_per_window_ * max_poas_);
        graph_details_d->nodes_to_visit = reinterpret_cast<decltype(graph_details_d->nodes_to_visit)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->nodes_to_visit) * max_nodes_per_window_ * max_poas_);
        graph_details_d->node_coverage_counts = reinterpret_cast<decltype(graph_details_d->node_coverage_counts)>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->node_coverage_counts) * max_nodes_per_window_ * max_poas_);
        if (output_mask_ & OutputType::msa)
        {
            graph_details_d->outgoing_edges_coverage = reinterpret_cast<decltype(graph_details_d->outgoing_edges_coverage)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->outgoing_edges_coverage) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa_ * max_poas_);
            graph_details_d->outgoing_edges_coverage_count = reinterpret_cast<decltype(graph_details_d->outgoing_edges_coverage_count)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->outgoing_edges_coverage_count) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
            graph_details_d->node_id_to_msa_pos = reinterpret_cast<decltype(graph_details_d->node_id_to_msa_pos)>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(*graph_details_d->node_id_to_msa_pos) * max_nodes_per_window_ * max_poas_);
        }

        *graph_details_d_p = graph_details_d;
        *graph_details_h_p = graph_details_h;
    }

    uint8_t* get_block_host()
    {
        return block_data_h_;
    }

    uint8_t* get_block_device()
    {
        return block_data_d_;
    }

    int32_t get_max_poas() const { return max_poas_; };

    static int64_t compute_device_memory_per_poa(const BatchConfig& batch_size, const bool msa_flag, const bool variable_bands = false)
    {
        int64_t device_size_per_poa = 0;
        int32_t max_nodes_per_graph = batch_size.max_nodes_per_graph;

        // for output - device
        device_size_per_poa += batch_size.max_consensus_size * sizeof(*OutputDetails::consensus);                                                                        // output_details_d_->consensus
        device_size_per_poa += (!msa_flag) ? batch_size.max_consensus_size * sizeof(*OutputDetails::coverage) : 0;                                                       // output_details_d_->coverage
        device_size_per_poa += (msa_flag) ? batch_size.max_consensus_size * batch_size.max_sequences_per_poa * sizeof(*OutputDetails::multiple_sequence_alignments) : 0; // output_details_d_->multiple_sequence_alignments
        // for input - device
        device_size_per_poa += batch_size.max_sequences_per_poa * batch_size.max_sequence_size * sizeof(*InputDetails<SizeT>::sequences);    // input_details_d_->sequences
        device_size_per_poa += batch_size.max_sequences_per_poa * batch_size.max_sequence_size * sizeof(*InputDetails<SizeT>::base_weights); // input_details_d_->base_weights
        device_size_per_poa += batch_size.max_sequences_per_poa * sizeof(*InputDetails<SizeT>::sequence_lengths);                            // input_details_d_->sequence_lengths
        device_size_per_poa += sizeof(*InputDetails<SizeT>::window_details);                                                                 // input_details_d_->window_details
        device_size_per_poa += (msa_flag) ? batch_size.max_sequences_per_poa * sizeof(*InputDetails<SizeT>::sequence_begin_nodes_ids) : 0;   // input_details_d_->sequence_begin_nodes_ids
        // for graph - device
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::nodes) * max_nodes_per_graph;                                                                                                // graph_details_d_->nodes
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::node_alignments) * max_nodes_per_graph * CUDAPOA_MAX_NODE_ALIGNMENTS;                                                        // graph_details_d_->node_alignments
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::node_alignment_count) * max_nodes_per_graph;                                                                                 // graph_details_d_->node_alignment_count
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::incoming_edges) * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES;                                                              // graph_details_d_->incoming_edges
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::incoming_edge_count) * max_nodes_per_graph;                                                                                  // graph_details_d_->incoming_edge_count
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::outgoing_edges) * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES;                                                              // graph_details_d_->outgoing_edges
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::outgoing_edge_count) * max_nodes_per_graph;                                                                                  // graph_details_d_->outgoing_edge_count
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::incoming_edge_weights) * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES;                                                       // graph_details_d_->incoming_edge_weights
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::outgoing_edge_weights) * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES;                                                       // graph_details_d_->outgoing_edge_weights
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::sorted_poa) * max_nodes_per_graph;                                                                                           // graph_details_d_->sorted_poa
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::sorted_poa_node_map) * max_nodes_per_graph;                                                                                  // graph_details_d_->sorted_poa_node_map
        device_size_per_poa += variable_bands ? sizeof(*GraphDetails<SizeT>::node_distance_to_head) * max_nodes_per_graph : 0;                                                           // graph_details_d_->node_distance_to_head
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::sorted_poa_local_edge_count) * max_nodes_per_graph;                                                                          // graph_details_d_->sorted_poa_local_edge_count
        device_size_per_poa += (!msa_flag) ? sizeof(*GraphDetails<SizeT>::consensus_scores) * max_nodes_per_graph : 0;                                                                   // graph_details_d_->consensus_scores
        device_size_per_poa += (!msa_flag) ? sizeof(*GraphDetails<SizeT>::consensus_predecessors) * max_nodes_per_graph : 0;                                                             // graph_details_d_->consensus_predecessors
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::node_marks) * max_nodes_per_graph;                                                                                           // graph_details_d_->node_marks
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::check_aligned_nodes) * max_nodes_per_graph;                                                                                  // graph_details_d_->check_aligned_nodes
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::nodes_to_visit) * max_nodes_per_graph;                                                                                       // graph_details_d_->nodes_to_visit
        device_size_per_poa += sizeof(*GraphDetails<SizeT>::node_coverage_counts) * max_nodes_per_graph;                                                                                 // graph_details_d_->node_coverage_counts
        device_size_per_poa += (msa_flag) ? sizeof(*GraphDetails<SizeT>::outgoing_edges_coverage) * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES * batch_size.max_sequences_per_poa : 0; // graph_details_d_->outgoing_edges_coverage
        device_size_per_poa += (msa_flag) ? sizeof(*GraphDetails<SizeT>::outgoing_edges_coverage_count) * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES : 0;                              // graph_details_d_->outgoing_edges_coverage_count
        device_size_per_poa += (msa_flag) ? sizeof(*GraphDetails<SizeT>::node_id_to_msa_pos) * max_nodes_per_graph : 0;                                                                  // graph_details_d_->node_id_to_msa_pos
        // for alignment - device
        device_size_per_poa += sizeof(*AlignmentDetails<ScoreT, SizeT>::alignment_graph) * max_nodes_per_graph;                        // alignment_details_d_->alignment_graph
        device_size_per_poa += sizeof(*AlignmentDetails<ScoreT, SizeT>::alignment_read) * max_nodes_per_graph;                         // alignment_details_d_->alignment_read
        device_size_per_poa += variable_bands ? sizeof(*AlignmentDetails<ScoreT, SizeT>::band_starts) * max_nodes_per_graph : 0;       // alignment_details_d_->band_starts
        device_size_per_poa += variable_bands ? sizeof(*AlignmentDetails<ScoreT, SizeT>::band_widths) * max_nodes_per_graph : 0;       // alignment_details_d_->band_widths
        device_size_per_poa += variable_bands ? sizeof(*AlignmentDetails<ScoreT, SizeT>::band_head_indices) * max_nodes_per_graph : 0; // alignment_details_d_->band_head_indices
        device_size_per_poa += variable_bands ? sizeof(*AlignmentDetails<ScoreT, SizeT>::band_max_indices) * max_nodes_per_graph : 0;  // alignment_details_d_->band_max_indices

        return device_size_per_poa;
    }

    static int64_t compute_host_memory_per_poa(const BatchConfig& batch_size, const bool msa_flag)
    {
        int64_t host_size_per_poa   = 0;
        int32_t max_nodes_per_graph = batch_size.max_nodes_per_graph;

        // for output - host
        host_size_per_poa += batch_size.max_consensus_size * sizeof(*OutputDetails::consensus);                                                                        // output_details_h_->consensus
        host_size_per_poa += (!msa_flag) ? batch_size.max_consensus_size * sizeof(*OutputDetails::coverage) : 0;                                                       // output_details_h_->coverage
        host_size_per_poa += (msa_flag) ? batch_size.max_consensus_size * batch_size.max_sequences_per_poa * sizeof(*OutputDetails::multiple_sequence_alignments) : 0; // output_details_h_->multiple_sequence_alignments
        host_size_per_poa += sizeof(OutputDetails);                                                                                                                    // output_details_d_
        // for input - host
        host_size_per_poa += batch_size.max_sequences_per_poa * batch_size.max_sequence_size * sizeof(*InputDetails<SizeT>::sequences);    // input_details_h_->sequences
        host_size_per_poa += batch_size.max_sequences_per_poa * batch_size.max_sequence_size * sizeof(*InputDetails<SizeT>::base_weights); // input_details_h_->base_weights
        host_size_per_poa += batch_size.max_sequences_per_poa * sizeof(*InputDetails<SizeT>::sequence_lengths);                            // input_details_h_->sequence_lengths
        host_size_per_poa += sizeof(*InputDetails<SizeT>::window_details);                                                                 // input_details_h_->window_details
        host_size_per_poa += (msa_flag) ? batch_size.max_sequences_per_poa * sizeof(*InputDetails<SizeT>::sequence_begin_nodes_ids) : 0;   // input_details_h_->sequence_begin_nodes_ids
        // for graph - host
        host_size_per_poa += sizeof(*GraphDetails<SizeT>::nodes) * max_nodes_per_graph;                                          // graph_details_h_->nodes
        host_size_per_poa += sizeof(*GraphDetails<SizeT>::incoming_edges) * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES;        // graph_details_d_->incoming_edges
        host_size_per_poa += sizeof(*GraphDetails<SizeT>::incoming_edge_weights) * max_nodes_per_graph * CUDAPOA_MAX_NODE_EDGES; // graph_details_d_->incoming_edge_weights
        host_size_per_poa += sizeof(*GraphDetails<SizeT>::incoming_edge_count) * max_nodes_per_graph;                            // graph_details_d_->incoming_edge_count

        return host_size_per_poa;
    }

    static int64_t estimate_max_poas(const BatchConfig& batch_size, bool msa_flag, float memory_usage_quota,
                                     int32_t mismatch_score, int32_t gap_score, int32_t match_score)
    {
        size_t total = 0, free = 0;
        cudaMemGetInfo(&free, &total);
        size_t mem_per_batch = memory_usage_quota * free; // Using memory_usage_quota of GPU available memory for cudapoa batch.

        int64_t sizeof_ScoreT       = 2;
        int64_t device_size_per_poa = 0;

        if (use32bitScore(batch_size, gap_score, mismatch_score, match_score))
        {
            sizeof_ScoreT = 4;
            if (use32bitSize(batch_size))
            {
                device_size_per_poa = BatchBlock<int32_t, int32_t>::compute_device_memory_per_poa(batch_size, msa_flag);
            }
            else
            {
                device_size_per_poa = BatchBlock<int32_t, int16_t>::compute_device_memory_per_poa(batch_size, msa_flag);
            }
        }
        else
        {
            // if ScoreT is 16-bit, it's safe to assume SizeT is also 16-bit
            device_size_per_poa = BatchBlock<int16_t, int16_t>::compute_device_memory_per_poa(batch_size, msa_flag);
        }

        // Compute required memory for score matrix
        int64_t device_size_per_score_matrix = static_cast<int64_t>(batch_size.matrix_sequence_dimension) *
                                               static_cast<int64_t>(batch_size.matrix_graph_dimension) * sizeof_ScoreT;

        // Calculate max POAs possible based on available memory.
        int64_t max_poas = mem_per_batch / (device_size_per_poa + device_size_per_score_matrix);

        return max_poas;
    }

protected:
    // Returns amount of host and device memory needed to store metadata per POA entry.
    // The first two elements of the tuple are fixed host and device sizes that
    // don't vary based on POA count. The latter two are host and device
    // buffer sizes that scale with number of POA entries to process. These sizes do
    // not include the scoring matrix needs for POA processing.
    std::tuple<int64_t, int64_t, int64_t, int64_t> calculate_space_per_poa(const BatchConfig& batch_size)
    {
        int64_t host_size_per_poa   = compute_host_memory_per_poa(batch_size, (output_mask_ & OutputType::msa));
        int64_t device_size_per_poa = compute_device_memory_per_poa(batch_size, (output_mask_ & OutputType::msa), variable_bands_);
        int64_t device_size_fixed   = 0;
        int64_t host_size_fixed     = 0;
        // for output - host
        host_size_fixed += sizeof(OutputDetails); // output_details_h_
        // for input - host
        host_size_fixed += sizeof(InputDetails<SizeT>); // input_details_h_
        host_size_fixed += sizeof(InputDetails<SizeT>); // input_details_d_
        // for graph - host
        host_size_fixed += sizeof(GraphDetails<SizeT>); // graph_details_h_
        host_size_fixed += sizeof(GraphDetails<SizeT>); // graph_details_d_
        // for alignment - host
        host_size_fixed += sizeof(AlignmentDetails<ScoreT, SizeT>); // alignment_details_d_

        return std::make_tuple(host_size_fixed, device_size_fixed, host_size_per_poa, device_size_per_poa);
    }

protected:
    // Maximum POAs to process in batch.
    int32_t max_poas_ = 0;

    // Maximum sequences per POA.
    int32_t max_sequences_per_poa_ = 0;

    // flag that enables some extra buffers to accommodate fully adaptive bands with variable width and arbitrary location
    // disabled for current implementation, can be enabled for possible future variants of adaptive alignment algorithm
    bool variable_bands_ = false;

    // Pointer for block data on host and device
    uint8_t* block_data_h_;
    uint8_t* block_data_d_;

    // Accumulator for the memory size
    int64_t total_h_ = 0;
    int64_t total_d_ = 0;

    // Offset index for pointing a buffer to block memory
    int64_t offset_h_ = 0;
    int64_t offset_d_ = 0;

    int64_t input_size_           = 0;
    int64_t output_size_          = 0;
    int32_t max_nodes_per_window_ = 0;
    int32_t device_id_;

    // Bit field for output type
    int8_t output_mask_;
};

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
