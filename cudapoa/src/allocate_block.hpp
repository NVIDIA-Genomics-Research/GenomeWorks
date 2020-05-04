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

#include "cudapoa_kernels.cuh"

#include <memory>
#include <vector>
#include <stdint.h>
#include <string>
#include <cuda_runtime_api.h>
#include <claragenomics/cudapoa/batch.hpp>

#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/logging/logging.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

#ifndef CGA_LOG_LEVEL
#ifndef NDEBUG
/// \brief Defines the logging level used in the current module
#define CGA_LOG_LEVEL cga_log_level_debug
#else // NDEBUG
/// \brief Defines the logging level used in the current module
#define CGA_LOG_LEVEL cga_log_level_error
#endif // NDEBUG
#endif // CGA_LOG_LEVEL

namespace claragenomics
{

namespace cudapoa
{

template <typename ScoreT, typename SizeT>
class BatchBlock
{
public:
    BatchBlock(int32_t device_id, size_t avail_mem, int8_t output_mask, const BatchSize& batch_size, bool banded_alignment = false)
        : max_sequences_per_poa_(throw_on_negative(batch_size.max_sequences_per_poa, "Maximum sequences per POA has to be non-negative"))
        , banded_alignment_(banded_alignment)
        , device_id_(throw_on_negative(device_id, "Device ID has to be non-negative"))
        , output_mask_(output_mask)
    {
        scoped_device_switch dev(device_id_);

        matrix_sequence_dimension_ = banded_alignment_ ? CUDAPOA_BANDED_MAX_MATRIX_SEQUENCE_DIMENSION : batch_size.max_matrix_sequence_dimension;
        max_graph_dimension_       = banded_alignment_ ? batch_size.max_matrix_graph_dimension_banded : batch_size.max_matrix_graph_dimension;
        max_nodes_per_window_      = banded_alignment_ ? batch_size.max_nodes_per_window_banded : batch_size.max_nodes_per_window;

        // calculate static and dynamic sizes of buffers needed per POA entry.
        int64_t host_size_fixed, device_size_fixed;
        int64_t host_size_per_poa, device_size_per_poa;
        std::tie(host_size_fixed, device_size_fixed, host_size_per_poa, device_size_per_poa) = calculate_space_per_poa(batch_size);

        // Using 2x as a buffer.
        size_t minimum_device_mem = 2 * (device_size_fixed + device_size_per_poa);
        if (avail_mem < minimum_device_mem)
        {
            std::string msg = std::string("Require at least ")
                                  .append(std::to_string(minimum_device_mem))
                                  .append(" bytes of device memory per CUDAPOA batch to process correctly.");
            throw std::runtime_error(msg);
        }

        // Calculate max POAs possible based on available memory.
        int64_t device_size_per_score_matrix = static_cast<int64_t>(matrix_sequence_dimension_) * static_cast<int64_t>(max_graph_dimension_) * sizeof(ScoreT);
        max_poas_                            = avail_mem / (device_size_per_poa + device_size_per_score_matrix);

        // Update final sizes for block based on calculated maximum POAs.
        output_size_ = max_poas_ * static_cast<int64_t>(batch_size.max_concensus_size);
        input_size_  = max_poas_ * max_sequences_per_poa_ * static_cast<int64_t>(batch_size.max_sequence_size);
        total_h_     = max_poas_ * host_size_per_poa + host_size_fixed;
        total_d_     = avail_mem;

        // Allocate.
        CGA_CU_CHECK_ERR(cudaHostAlloc((void**)&block_data_h_, total_h_, cudaHostAllocDefault));
        CGA_CU_CHECK_ERR(cudaMalloc((void**)&block_data_d_, total_d_));
    }

    ~BatchBlock()
    {
        CGA_CU_CHECK_ERR(cudaFree(block_data_d_));
        CGA_CU_CHECK_ERR(cudaFreeHost(block_data_h_));
    }

    void get_output_details(OutputDetails** output_details_h_p, OutputDetails** output_details_d_p)
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
        offset_d_ += cudautils::align<int64_t, 8>(output_size_ * sizeof(int8_t));
        if (output_mask_ & OutputType::consensus)
        {
            output_details_d->coverage = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(output_size_ * sizeof(int16_t));
        }
        if (output_mask_ & OutputType::msa)
        {
            output_details_d->multiple_sequence_alignments = reinterpret_cast<uint8_t*>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(output_size_ * max_sequences_per_poa_ * sizeof(uint8_t));
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
        offset_h_ += input_size_ * sizeof(uint8_t);
        input_details_h->base_weights = reinterpret_cast<int8_t*>(&block_data_h_[offset_h_]);
        offset_h_ += input_size_ * sizeof(int8_t);
        input_details_h->sequence_lengths = reinterpret_cast<SizeT*>(&block_data_h_[offset_h_]);
        offset_h_ += max_poas_ * max_sequences_per_poa_ * sizeof(SizeT);
        input_details_h->window_details = reinterpret_cast<WindowDetails*>(&block_data_h_[offset_h_]);
        offset_h_ += max_poas_ * sizeof(WindowDetails);
        if (output_mask_ & OutputType::msa)
        {
            input_details_h->sequence_begin_nodes_ids = reinterpret_cast<SizeT*>(&block_data_h_[offset_h_]);
            offset_h_ += max_poas_ * max_sequences_per_poa_ * sizeof(SizeT);
        }

        input_details_d = reinterpret_cast<InputDetails<SizeT>*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(InputDetails<SizeT>);

        // on device
        input_details_d->sequences = &block_data_d_[offset_d_];
        offset_d_ += cudautils::align<int64_t, 8>(input_size_ * sizeof(uint8_t));
        input_details_d->base_weights = reinterpret_cast<int8_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(input_size_ * sizeof(int8_t));
        input_details_d->sequence_lengths = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(max_poas_ * max_sequences_per_poa_ * sizeof(SizeT));
        input_details_d->window_details = reinterpret_cast<WindowDetails*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(max_poas_ * sizeof(WindowDetails));
        if (output_mask_ & OutputType::msa)
        {
            input_details_d->sequence_begin_nodes_ids = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(max_poas_ * max_sequences_per_poa_ * sizeof(SizeT));
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
        alignment_details_d->alignment_graph = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_graph_dimension_ * max_poas_);
        alignment_details_d->alignment_read = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_graph_dimension_ * max_poas_);

        // rest of the available memory is assigned to scores buffer
        alignment_details_d->scorebuf_alloc_size = total_d_ - offset_d_;
        alignment_details_d->scores              = reinterpret_cast<ScoreT*>(&block_data_d_[offset_d_]);
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
        offset_h_ += sizeof(uint8_t) * max_nodes_per_window_ * max_poas_;
        graph_details_h->incoming_edges = reinterpret_cast<SizeT*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(SizeT) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
        graph_details_h->incoming_edge_weights = reinterpret_cast<uint16_t*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_;
        graph_details_h->incoming_edge_count = reinterpret_cast<uint16_t*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(uint16_t) * max_nodes_per_window_ * max_poas_;
        graph_details_d = reinterpret_cast<GraphDetails<SizeT>*>(&block_data_h_[offset_h_]);
        offset_h_ += sizeof(GraphDetails<SizeT>);
        graph_details_d->nodes = &block_data_h_[offset_h_];

        // on device
        graph_details_d->nodes = &block_data_d_[offset_d_];
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint8_t) * max_nodes_per_window_ * max_poas_);
        graph_details_d->node_alignments = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_ALIGNMENTS * max_poas_);
        graph_details_d->node_alignment_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * max_poas_);
        graph_details_d->incoming_edges = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
        graph_details_d->incoming_edge_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * max_poas_);
        graph_details_d->outgoing_edges = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
        graph_details_d->outgoing_edge_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * max_poas_);
        graph_details_d->incoming_edge_weights = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
        graph_details_d->outgoing_edge_weights = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
        graph_details_d->sorted_poa = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_nodes_per_window_ * max_poas_);
        graph_details_d->sorted_poa_node_map = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_nodes_per_window_ * max_poas_);
        graph_details_d->sorted_poa_local_edge_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * max_poas_);
        if (output_mask_ & OutputType::consensus)
        {
            graph_details_d->consensus_scores = reinterpret_cast<int32_t*>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(int32_t) * max_nodes_per_window_ * max_poas_);
            graph_details_d->consensus_predecessors = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_nodes_per_window_ * max_poas_);
        }

        graph_details_d->node_marks = reinterpret_cast<uint8_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(int8_t) * max_nodes_per_window_ * max_poas_);
        graph_details_d->check_aligned_nodes = reinterpret_cast<bool*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(bool) * max_nodes_per_window_ * max_poas_);
        graph_details_d->nodes_to_visit = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_nodes_per_window_ * max_poas_);
        graph_details_d->node_coverage_counts = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
        offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * max_poas_);
        if (output_mask_ & OutputType::msa)
        {
            graph_details_d->outgoing_edges_coverage = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa_ * max_poas_);
            graph_details_d->outgoing_edges_coverage_count = reinterpret_cast<uint16_t*>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_poas_);
            graph_details_d->node_id_to_msa_pos = reinterpret_cast<SizeT*>(&block_data_d_[offset_d_]);
            offset_d_ += cudautils::align<int64_t, 8>(sizeof(SizeT) * max_nodes_per_window_ * max_poas_);
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

protected:
    // Returns amount of host and device memory needed to store metadata per POA entry.
    // The first two elements of the tuple are fixed host and device sizes that
    // don't vary based on POA count. The latter two are host and device
    // buffer sizes that scale with number of POA entries to process. These sizes do
    // not include the scoring matrix needs for POA processing.
    std::tuple<int64_t, int64_t, int64_t, int64_t> calculate_space_per_poa(const BatchSize& batch_size)
    {
        const int32_t poa_count = 1;

        int64_t host_size_fixed = 0, device_size_fixed = 0;
        int64_t host_size_per_poa = 0, device_size_per_poa = 0;

        int64_t input_size_per_poa  = max_sequences_per_poa_ * batch_size.max_sequence_size * poa_count;
        int64_t output_size_per_poa = batch_size.max_concensus_size * poa_count;

        // for output - host
        host_size_fixed += sizeof(OutputDetails);                                                                                   // output_details_h_
        host_size_per_poa += output_size_per_poa * sizeof(uint8_t);                                                                 // output_details_h_->consensus
        host_size_per_poa += (output_mask_ & OutputType::consensus) ? output_size_per_poa * sizeof(uint16_t) : 0;                   // output_details_h_->coverage
        host_size_per_poa += (output_mask_ & OutputType::msa) ? output_size_per_poa * max_sequences_per_poa_ * sizeof(uint8_t) : 0; // output_details_h_->multiple_sequence_alignments
        host_size_per_poa += sizeof(OutputDetails);                                                                                 // output_details_d_
        // for output - device
        device_size_per_poa += output_size_per_poa * sizeof(uint8_t);                                                                 // output_details_d_->consensus
        device_size_per_poa += (output_mask_ & OutputType::consensus) ? output_size_per_poa * sizeof(uint16_t) : 0;                   // output_details_d_->coverage
        device_size_per_poa += (output_mask_ & OutputType::msa) ? output_size_per_poa * max_sequences_per_poa_ * sizeof(uint8_t) : 0; // output_details_d_->multiple_sequence_alignments

        // for input - host
        host_size_fixed += sizeof(InputDetails<SizeT>);                                                                 // input_details_h_
        host_size_per_poa += input_size_per_poa * sizeof(uint8_t);                                                      // input_details_h_->sequences
        host_size_per_poa += input_size_per_poa * sizeof(int8_t);                                                       // input_details_h_->base_weights
        host_size_per_poa += poa_count * max_sequences_per_poa_ * sizeof(SizeT);                                        // input_details_h_->sequence_lengths
        host_size_per_poa += poa_count * sizeof(WindowDetails);                                                         // input_details_h_->window_details
        host_size_per_poa += (output_mask_ & OutputType::msa) ? poa_count * max_sequences_per_poa_ * sizeof(SizeT) : 0; // input_details_h_->sequence_begin_nodes_ids

        host_size_fixed += sizeof(InputDetails<SizeT>); // input_details_d_
        // for input - device
        device_size_per_poa += input_size_per_poa * sizeof(uint8_t);                                                      // input_details_d_->sequences
        device_size_per_poa += input_size_per_poa * sizeof(int8_t);                                                       // input_details_d_->base_weights
        device_size_per_poa += poa_count * max_sequences_per_poa_ * sizeof(SizeT);                                        // input_details_d_->sequence_lengths
        device_size_per_poa += poa_count * sizeof(WindowDetails);                                                         // input_details_d_->window_details
        device_size_per_poa += (output_mask_ & OutputType::msa) ? poa_count * max_sequences_per_poa_ * sizeof(SizeT) : 0; // input_details_d_->sequence_begin_nodes_ids

        // for graph - host
        host_size_fixed += sizeof(GraphDetails<SizeT>);                                                     // graph_details_h_
        host_size_fixed += sizeof(GraphDetails<SizeT>);                                                     // graph_details_d_
        host_size_per_poa += sizeof(uint8_t) * max_nodes_per_window_ * poa_count;                           // graph_details_h_->nodes
        host_size_per_poa += sizeof(SizeT) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * poa_count;    // graph_details_d_->incoming_edges
        host_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * poa_count; // graph_details_d_->incoming_edge_weights
        host_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * poa_count;                          // graph_details_d_->incoming_edge_count

        // for graph - device
        device_size_per_poa += sizeof(uint8_t) * max_nodes_per_window_ * poa_count;                                                                                           // graph_details_d_->nodes
        device_size_per_poa += sizeof(SizeT) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_ALIGNMENTS * poa_count;                                                               // graph_details_d_->node_alignments
        device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * poa_count;                                                                                          // graph_details_d_->node_alignment_count
        device_size_per_poa += sizeof(SizeT) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * poa_count;                                                                    // graph_details_d_->incoming_edges
        device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * poa_count;                                                                                          // graph_details_d_->incoming_edge_count
        device_size_per_poa += sizeof(SizeT) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * poa_count;                                                                    // graph_details_d_->outgoing_edges
        device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * poa_count;                                                                                          // graph_details_d_->outgoing_edge_count
        device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * poa_count;                                                                 // graph_details_d_->incoming_edge_weights
        device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * poa_count;                                                                 // graph_details_d_->outgoing_edge_weights
        device_size_per_poa += sizeof(SizeT) * max_nodes_per_window_ * poa_count;                                                                                             // graph_details_d_->sorted_poa
        device_size_per_poa += sizeof(SizeT) * max_nodes_per_window_ * poa_count;                                                                                             // graph_details_d_->sorted_poa_node_map
        device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * poa_count;                                                                                          // graph_details_d_->sorted_poa_local_edge_count
        device_size_per_poa += (output_mask_ & OutputType::consensus) ? sizeof(int32_t) * max_nodes_per_window_ * poa_count : 0;                                              // graph_details_d_->consensus_scores
        device_size_per_poa += (output_mask_ & OutputType::consensus) ? sizeof(SizeT) * max_nodes_per_window_ * poa_count : 0;                                                // graph_details_d_->consensus_predecessors
        device_size_per_poa += sizeof(int8_t) * max_nodes_per_window_ * poa_count;                                                                                            // graph_details_d_->node_marks
        device_size_per_poa += sizeof(bool) * max_nodes_per_window_ * poa_count;                                                                                              // graph_details_d_->check_aligned_nodes
        device_size_per_poa += sizeof(SizeT) * max_nodes_per_window_ * poa_count;                                                                                             // graph_details_d_->nodes_to_visit
        device_size_per_poa += sizeof(uint16_t) * max_nodes_per_window_ * poa_count;                                                                                          // graph_details_d_->node_coverage_counts
        device_size_per_poa += (output_mask_ & OutputType::msa) ? sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa_ * poa_count : 0; // graph_details_d_->outgoing_edges_coverage
        device_size_per_poa += (output_mask_ & OutputType::msa) ? sizeof(uint16_t) * max_nodes_per_window_ * CUDAPOA_MAX_NODE_EDGES * poa_count : 0;                          // graph_details_d_->outgoing_edges_coverage_count
        device_size_per_poa += (output_mask_ & OutputType::msa) ? sizeof(SizeT) * max_nodes_per_window_ * poa_count : 0;                                                      // graph_details_d_->node_id_to_msa_pos

        // for alignment - host
        host_size_fixed += sizeof(AlignmentDetails<ScoreT, SizeT>); // alignment_details_d_
        // for alignment - device
        device_size_per_poa += sizeof(SizeT) * max_graph_dimension_ * poa_count; // alignment_details_d_->alignment_graph
        device_size_per_poa += sizeof(SizeT) * max_graph_dimension_ * poa_count; // alignment_details_d_->alignment_read

        return std::make_tuple(host_size_fixed, device_size_fixed, host_size_per_poa, device_size_per_poa);
    }

protected:
    // Maximum POAs to process in batch.
    int32_t max_poas_ = 0;

    // Maximum sequences per POA.
    int32_t max_sequences_per_poa_ = 0;

    // Use banded POA alignment
    bool banded_alignment_;

    // Pointer for block data on host and device
    uint8_t* block_data_h_;
    uint8_t* block_data_d_;

    // Accumulator for the memory size
    int64_t total_h_ = 0;
    int64_t total_d_ = 0;

    // Offset index for pointing a buffer to block memory
    int64_t offset_h_ = 0;
    int64_t offset_d_ = 0;

    int64_t input_size_                = 0;
    int64_t output_size_               = 0;
    int32_t matrix_sequence_dimension_ = 0;
    int32_t max_graph_dimension_       = 0;
    int32_t max_nodes_per_window_      = 0;
    int32_t device_id_;

    // Bit field for output type
    int8_t output_mask_;
};

} // namespace cudapoa

} // namespace claragenomics
