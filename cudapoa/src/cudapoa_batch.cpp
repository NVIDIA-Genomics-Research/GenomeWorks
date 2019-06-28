/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
#include <cstring>

#include "allocate_block.hpp"
#include "cudapoa_batch.hpp"
#include "cudapoa_kernels.cuh"

#include <cudautils/cudautils.hpp>
#include <logging/logging.hpp>
#include <utils/signed_integer_utils.hpp>

#ifndef TABS
#define TABS printTabs(bid_)
#endif

inline std::string printTabs(int32_t tab_count)
{
    std::string s;
    for (int32_t i = 0; i < tab_count; i++)
    {
        s += "\t";
    }
    return s;
}

namespace cga
{

namespace cudapoa
{

int32_t CudapoaBatch::batches = 0;

void CudapoaBatch::print_batch_debug_message(const std::string& message)
{
    (void)message;
    CGA_LOG_DEBUG("{}{}{}{}", TABS, bid_, message, device_id_);
}

void CudapoaBatch::initialize_output_details()
{
    batch_block_->get_output_details(&output_details_h_, &output_details_d_);
}

void CudapoaBatch::initialize_input_details()
{
    batch_block_->get_input_details(&input_details_h_, &input_details_d_);
}

void CudapoaBatch::initialize_alignment_details()
{
    batch_block_->get_alignment_details(&alignment_details_d_);
}

void CudapoaBatch::initialize_graph_details()
{
    batch_block_->get_graph_details(&graph_details_d_);
}

CudapoaBatch::CudapoaBatch(int32_t max_poas, int32_t max_sequences_per_poa, int32_t device_id, int8_t output_mask, int16_t gap_score, int16_t mismatch_score, int16_t match_score, bool cuda_banded_alignment)
    : max_poas_(throw_on_negative(max_poas, "Maximum POAs in batch has to be non-negative"))
    , max_sequences_per_poa_(throw_on_negative(max_sequences_per_poa, "Maximum sequences per POA has to be non-negative"))
    , device_id_(throw_on_negative(device_id, "Device ID has to be non-negative"))
    , output_mask_(output_mask)
    , gap_score_(gap_score)
    , mismatch_score_(mismatch_score)
    , match_score_(match_score)
    , banded_alignment_(cuda_banded_alignment)
    , batch_block_(new BatchBlock(device_id, max_poas, max_sequences_per_poa, output_mask, cuda_banded_alignment))
{
    bid_ = CudapoaBatch::batches++;

    // Set CUDA device
    CGA_CU_CHECK_ERR(cudaSetDevice(device_id_));
    std::string msg = " Initializing batch on device ";
    print_batch_debug_message(msg);

    // Allocate host memory and CUDA memory based on max sequence and target counts.

    // Verify that maximum sequence size is in multiples of tb size.
    // We subtract one because the matrix dimension needs to be one element larger
    // than the sequence size.
    if (CUDAPOA_MAX_SEQUENCE_SIZE % CUDAPOA_THREADS_PER_BLOCK != 0)
    {
        CGA_LOG_CRITICAL("Thread block size needs to be in multiples of 32.");
        exit(-1);
    }

    initialize_input_details();

    int32_t input_size = max_poas_ * max_sequences_per_poa_ * CUDAPOA_MAX_SEQUENCE_SIZE; //TODO how big does this need to be
    msg                = " Allocated input buffers of size " + std::to_string((static_cast<float>(input_size) / (1024 * 1024))) + "MB on device ";
    print_batch_debug_message(msg);

    initialize_output_details();

    input_size += input_size * sizeof(uint16_t);
    msg = " Allocated output buffers of size " + std::to_string((static_cast<float>(input_size) / (1024 * 1024))) + "MB on device ";
    print_batch_debug_message(msg);

    initialize_alignment_details();
    initialize_graph_details();

    // Debug print for size allocated.
    int32_t matrix_sequence_dimension = banded_alignment_ ? CUDAPOA_BANDED_MAX_MATRIX_SEQUENCE_DIMENSION : CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION;
    int32_t max_graph_dimension       = cuda_banded_alignment ? CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION_BANDED : CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION;

    int32_t temp_size = (sizeof(int16_t) * max_graph_dimension * matrix_sequence_dimension * max_poas_);

    temp_size += 2 * (sizeof(int16_t) * max_graph_dimension * max_poas_);
    msg = " Allocated temp buffers of size " + std::to_string((static_cast<float>(temp_size) / (1024 * 1024))) + "MB on device ";
    print_batch_debug_message(msg);

    // Debug print for size allocated.
    uint16_t max_nodes_per_window = banded_alignment_ ? CUDAPOA_MAX_NODES_PER_WINDOW_BANDED : CUDAPOA_MAX_NODES_PER_WINDOW;
    temp_size                     = sizeof(uint8_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_ALIGNMENTS * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(int16_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(int16_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(int8_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(bool) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * max_poas_;
    temp_size += sizeof(uint16_t) * max_nodes_per_window * max_poas_;
    msg = " Allocated temp buffers of size " + std::to_string((static_cast<float>(temp_size) / (1024 * 1024))) + "MB on device ";
    print_batch_debug_message(msg);
}

CudapoaBatch::~CudapoaBatch()
{
    std::string msg = "Destroyed buffers on device ";
    print_batch_debug_message(msg);
}

int32_t CudapoaBatch::batch_id() const
{
    return bid_;
}

int32_t CudapoaBatch::get_total_poas() const
{
    return poa_count_;
}

void CudapoaBatch::generate_poa()
{
    CGA_CU_CHECK_ERR(cudaSetDevice(device_id_));
    //Copy sequencecs, sequence lengths and window details to device
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(input_details_d_->sequences, input_details_h_->sequences,
                                     num_nucleotides_copied_ * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_));
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(input_details_d_->base_weights, input_details_h_->base_weights,
                                     num_nucleotides_copied_ * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_));
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(input_details_d_->window_details, input_details_h_->window_details,
                                     poa_count_ * sizeof(cga::cudapoa::WindowDetails), cudaMemcpyHostToDevice, stream_));
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(input_details_d_->sequence_lengths, input_details_h_->sequence_lengths,
                                     global_sequence_idx_ * sizeof(uint16_t), cudaMemcpyHostToDevice, stream_));

    // Launch kernel to run 1 POA per thread in thread block.
    std::string msg = " Launching kernel for " + std::to_string(poa_count_) + " on device ";
    print_batch_debug_message(msg);

    cga::cudapoa::generatePOA(output_details_d_,
                              input_details_d_,
                              poa_count_,
                              stream_,
                              alignment_details_d_,
                              graph_details_d_,
                              gap_score_,
                              mismatch_score_,
                              match_score_,
                              banded_alignment_,
                              max_sequences_per_poa_,
                              output_mask_);

    CGA_CU_CHECK_ERR(cudaPeekAtLastError());
    msg = " Launched kernel on device ";
    print_batch_debug_message(msg);
}

void CudapoaBatch::decode_cudapoa_kernel_error(cga::cudapoa::StatusType error_type,
                                               std::vector<StatusType>& output_status)
{
    switch (error_type)
    {
    case cga::cudapoa::StatusType::node_count_exceeded_maximum_graph_size:
        CGA_LOG_ERROR("Kernel Error:: Node count exceeded maximum nodes per window\n");
        output_status.emplace_back(cga::cudapoa::StatusType::node_count_exceeded_maximum_graph_size);
        break;
    case cga::cudapoa::StatusType::seq_len_exceeded_maximum_nodes_per_window:
        CGA_LOG_ERROR("Kernel Error::Sequence length exceeded maximum nodes per window\n");
        output_status.emplace_back(cga::cudapoa::StatusType::seq_len_exceeded_maximum_nodes_per_window);
        break;
    case cga::cudapoa::StatusType::loop_count_exceeded_upper_bound:
        CGA_LOG_ERROR("Kernel Error::Loop count exceeded upper bound in nw algorithm\n");
        output_status.emplace_back(cga::cudapoa::StatusType::loop_count_exceeded_upper_bound);
        break;
    default:
        break;
    }
}

void CudapoaBatch::get_consensus(std::vector<std::string>& consensus,
                                 std::vector<std::vector<uint16_t>>& coverage,
                                 std::vector<StatusType>& output_status)
{
    std::string msg = " Launching memcpy D2H on device ";
    print_batch_debug_message(msg);
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(output_details_h_->consensus,
                                     output_details_d_->consensus,
                                     CUDAPOA_MAX_CONSENSUS_SIZE * max_poas_ * sizeof(uint8_t),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(output_details_h_->coverage,
                                     output_details_d_->coverage,
                                     CUDAPOA_MAX_CONSENSUS_SIZE * max_poas_ * sizeof(uint16_t),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    CGA_CU_CHECK_ERR(cudaStreamSynchronize(stream_));

    msg = " Finished memcpy D2H on device ";
    print_batch_debug_message(msg);

    for (int32_t poa = 0; poa < poa_count_; poa++)
    {
        // Get the consensus string and reverse it since on GPU the
        // string is built backwards..
        char* c = reinterpret_cast<char*>(&(output_details_h_->consensus[poa * CUDAPOA_MAX_CONSENSUS_SIZE]));
        // We use the first two entries in the consensus buffer to log error during kernel execution
        // c[0] == 0 means an error occured and when that happens the error type is saved in c[1]
        if (static_cast<uint8_t>(c[0]) == CUDAPOA_KERNEL_ERROR_ENCOUNTERED)
        {
            decode_cudapoa_kernel_error(static_cast<cga::cudapoa::StatusType>(c[1]), output_status);
            // push back empty placeholder for consensus and coverage
            consensus.emplace_back(std::string());
            coverage.emplace_back(std::vector<uint16_t>());
        }
        else
        {
            output_status.emplace_back(cga::cudapoa::StatusType::success);
            consensus.emplace_back(std::string(c));
            std::reverse(consensus.back().begin(), consensus.back().end());
            // Similarly, get the coverage and reverse it.
            coverage.emplace_back(std::vector<uint16_t>(
                &(output_details_h_->coverage[poa * CUDAPOA_MAX_CONSENSUS_SIZE]),
                &(output_details_h_->coverage[poa * CUDAPOA_MAX_CONSENSUS_SIZE + get_size(consensus.back())])));
            std::reverse(coverage.back().begin(), coverage.back().end());
        }
    }
}

void CudapoaBatch::get_msa(std::vector<std::vector<std::string>>& msa, std::vector<StatusType>& output_status)
{
    std::string msg = " Launching memcpy D2H on device for msa ";
    print_batch_debug_message(msg);

    CGA_CU_CHECK_ERR(cudaMemcpyAsync(output_details_h_->multiple_sequence_alignments,
                                     output_details_d_->multiple_sequence_alignments,
                                     max_poas_ * max_sequences_per_poa_ * CUDAPOA_MAX_CONSENSUS_SIZE * sizeof(uint8_t),
                                     cudaMemcpyDeviceToHost,
                                     stream_));

    CGA_CU_CHECK_ERR(cudaMemcpyAsync(output_details_h_->consensus,
                                     output_details_d_->consensus,
                                     CUDAPOA_MAX_CONSENSUS_SIZE * max_poas_ * sizeof(uint8_t),
                                     cudaMemcpyDeviceToHost,
                                     stream_));

    CGA_CU_CHECK_ERR(cudaStreamSynchronize(stream_));

    msg = " Finished memcpy D2H on device for msa";
    print_batch_debug_message(msg);

    for (int32_t poa = 0; poa < poa_count_; poa++)
    {
        msa.emplace_back(std::vector<std::string>());
        char* c = reinterpret_cast<char*>(&(output_details_h_->consensus[poa * CUDAPOA_MAX_CONSENSUS_SIZE]));
        // We use the first two entries in the consensus buffer to log error during kernel execution
        // c[0] == 0 means an error occured and when that happens the error type is saved in c[1]
        if (static_cast<uint8_t>(c[0]) == CUDAPOA_KERNEL_ERROR_ENCOUNTERED)
        {
            decode_cudapoa_kernel_error(static_cast<cga::cudapoa::StatusType>(c[1]), output_status);
        }
        else
        {
            output_status.emplace_back(cga::cudapoa::StatusType::success);
            uint16_t num_seqs = input_details_h_->window_details[poa].num_seqs;
            for (uint16_t i = 0; i < num_seqs; i++)
            {
                char* c = reinterpret_cast<char*>(&(output_details_h_->multiple_sequence_alignments[(poa * max_sequences_per_poa_ + i) * CUDAPOA_MAX_CONSENSUS_SIZE]));
                msa[poa].emplace_back(std::string(c));
            }
        }
    }
}

void CudapoaBatch::set_cuda_stream(cudaStream_t stream)
{
    stream_ = stream;
}

StatusType CudapoaBatch::add_poa()
{
    if (poa_count_ == max_poas_)
    {
        return StatusType::exceeded_maximum_poas;
    }

    WindowDetails window_details{};
    window_details.seq_len_buffer_offset         = global_sequence_idx_;
    window_details.seq_starts                    = num_nucleotides_copied_;
    input_details_h_->window_details[poa_count_] = window_details;
    poa_count_++;

    return StatusType::success;
}

void CudapoaBatch::reset()
{
    poa_count_              = 0;
    num_nucleotides_copied_ = 0;
    global_sequence_idx_    = 0;
}

StatusType CudapoaBatch::add_seq_to_poa(const char* seq, const int8_t* weights, int32_t seq_len)
{
    if (seq_len >= CUDAPOA_MAX_SEQUENCE_SIZE)
    {
        return StatusType::exceeded_maximum_sequence_size;
    }

    WindowDetails* window_details = &(input_details_h_->window_details[poa_count_ - 1]);

    if (static_cast<int32_t>(window_details->num_seqs) + 1 >= max_sequences_per_poa_)
    {
        return StatusType::exceeded_maximum_sequences_per_poa;
    }

    window_details->num_seqs++;
    // Copy sequence data
    memcpy(&(input_details_h_->sequences[num_nucleotides_copied_]),
           seq,
           seq_len);
    // Copy weights
    if (weights == nullptr)
    {
        memset(&(input_details_h_->base_weights[num_nucleotides_copied_]),
               1,
               seq_len);
    }
    else
    {
        // Verify that weightsw are positive.
        for (int32_t i = 0; i < seq_len; i++)
        {
            throw_on_negative(weights[i], "Base weights need have to be non-negative");
        }
        memcpy(&(input_details_h_->base_weights[num_nucleotides_copied_]),
               weights,
               seq_len);
    }
    input_details_h_->sequence_lengths[global_sequence_idx_] = seq_len;

    num_nucleotides_copied_ += seq_len;
    global_sequence_idx_++;

    return StatusType::success;
}

} // namespace cudapoa

} // namespace cga
