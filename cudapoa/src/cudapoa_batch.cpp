#include <algorithm>
#include <cstring>

#define GW_LOG_LEVEL gw_log_level_info

#include "cudapoa/batch.hpp"
#include "cudapoa_batch.hpp"
#include "cudapoa_kernels.cuh"

#include <cudautils/cudautils.hpp>
#include <logging/logging.hpp>

#ifndef TABS
#define TABS printTabs(bid_)
#endif

inline std::string printTabs(uint32_t tab_count)
{
    std::string s;
    for(uint32_t i = 0; i < tab_count; i++)
    {
        s += "\t";
    }
    return s;
}

namespace genomeworks {

namespace cudapoa {

uint32_t CudapoaBatch::batches = 0;

void CudapoaBatch::print_batch_debug_message(const std::string& message)
{
    GW_LOG_INFO("{}{}{}{}", TABS, bid_, message, device_id_);
}

void CudapoaBatch::initialize_output_details()
{
    // Output buffers.
    uint32_t input_size = max_poas_ * CUDAPOA_MAX_SEQUENCE_SIZE;
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &output_details_h_, sizeof(genomeworks::cudapoa::OutputDetails), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &(output_details_h_->consensus), input_size * sizeof(uint8_t), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &(output_details_h_->coverage), input_size * sizeof(uint16_t), cudaHostAllocDefault));

    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &output_details_d_, sizeof(genomeworks::cudapoa::OutputDetails), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(output_details_d_->consensus), input_size * sizeof(int8_t)));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(output_details_d_->coverage), input_size * sizeof(int16_t)));
}

void CudapoaBatch::free_output_details()
{
    GW_CU_CHECK_ERR(cudaFreeHost(output_details_h_->consensus));
    GW_CU_CHECK_ERR(cudaFreeHost(output_details_h_->coverage));
    GW_CU_CHECK_ERR(cudaFreeHost(output_details_h_));
    GW_CU_CHECK_ERR(cudaFree(output_details_d_->consensus));
    GW_CU_CHECK_ERR(cudaFree(output_details_d_->coverage));
    GW_CU_CHECK_ERR(cudaFreeHost(output_details_d_));
}

void CudapoaBatch::initialize_input_details()
{
    uint32_t input_size = max_poas_ * max_sequences_per_poa_ * CUDAPOA_MAX_SEQUENCE_SIZE; //TODO how big does this need to be
    // Host allocations
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &input_details_h_, sizeof(genomeworks::cudapoa::InputDetails), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &(input_details_h_->sequences), input_size * sizeof(uint8_t), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &(input_details_h_->sequence_lengths), max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &(input_details_h_->window_details), max_poas_ * sizeof(WindowDetails), cudaHostAllocDefault));
    // Device allocations
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &input_details_d_, sizeof(genomeworks::cudapoa::InputDetails), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(input_details_d_->sequences), input_size * sizeof(uint8_t)));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(input_details_d_->sequence_lengths), max_poas_ * max_sequences_per_poa_ * sizeof(uint16_t)));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(input_details_d_->window_details), max_poas_ * sizeof(WindowDetails)));
}

void CudapoaBatch::free_input_details()
{
    GW_CU_CHECK_ERR(cudaFreeHost(input_details_h_->sequences));
    GW_CU_CHECK_ERR(cudaFreeHost(input_details_h_->sequence_lengths));
    GW_CU_CHECK_ERR(cudaFreeHost(input_details_h_->window_details));
    GW_CU_CHECK_ERR(cudaFreeHost(input_details_h_));
    GW_CU_CHECK_ERR(cudaFree(input_details_d_->sequences));
    GW_CU_CHECK_ERR(cudaFree(input_details_d_->sequence_lengths));
    GW_CU_CHECK_ERR(cudaFree(input_details_d_->window_details));
    GW_CU_CHECK_ERR(cudaFreeHost(input_details_d_));
}

void CudapoaBatch::initialize_alignment_details()
{
    // Struct for alignment details
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &alignment_details_d_, sizeof(genomeworks::cudapoa::AlignmentDetails), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(alignment_details_d_->scores), sizeof(int16_t) * CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION * max_poas_));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(alignment_details_d_->alignment_graph), sizeof(int16_t) * CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(alignment_details_d_->alignment_read), sizeof(int16_t) * CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION * max_poas_ ));
}

void CudapoaBatch::free_alignment_details()
{
    GW_CU_CHECK_ERR(cudaFree(alignment_details_d_->scores));
    GW_CU_CHECK_ERR(cudaFree(alignment_details_d_->alignment_graph));
    GW_CU_CHECK_ERR(cudaFree(alignment_details_d_->alignment_read));
    GW_CU_CHECK_ERR(cudaFreeHost(alignment_details_d_));
}

void CudapoaBatch::initialize_graph_details()
{
    // Struct for graph details
    GW_CU_CHECK_ERR(cudaHostAlloc((void**) &graph_details_d_, sizeof(genomeworks::cudapoa::GraphDetails), cudaHostAllocDefault));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->nodes), sizeof(uint8_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->node_alignments), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_ALIGNMENTS * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->node_alignment_count), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->incoming_edges), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->incoming_edge_count), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->outgoing_edges), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->outgoing_edge_count), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->incoming_edge_weights), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->outgoing_edge_weights), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->sorted_poa), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->sorted_poa_node_map), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->sorted_poa_local_edge_count), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->consensus_scores), sizeof(int32_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->consensus_predecessors), sizeof(int16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->node_marks), sizeof(int8_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->check_aligned_nodes), sizeof(bool) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->nodes_to_visit), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
    GW_CU_CHECK_ERR(cudaMalloc((void**) &(graph_details_d_->node_coverage_counts), sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ));
}

void CudapoaBatch::free_graph_details()
{
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->nodes));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->node_alignments));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->node_alignment_count));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->incoming_edges));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->incoming_edge_count));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->outgoing_edges));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->outgoing_edge_count));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->incoming_edge_weights));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->outgoing_edge_weights));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->sorted_poa));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->sorted_poa_node_map));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->sorted_poa_local_edge_count));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->consensus_scores));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->consensus_predecessors));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->node_marks));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->check_aligned_nodes));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->nodes_to_visit));
    GW_CU_CHECK_ERR(cudaFree(graph_details_d_->node_coverage_counts));
    GW_CU_CHECK_ERR(cudaFreeHost(graph_details_d_));
}

CudapoaBatch::CudapoaBatch(uint32_t max_poas, uint32_t max_sequences_per_poa, uint32_t device_id, int16_t gap_score, int16_t mismatch_score, int16_t match_score)
    : max_poas_(max_poas), max_sequences_per_poa_(max_sequences_per_poa), device_id_(device_id), gap_score_(gap_score), mismatch_score_(mismatch_score), match_score_(match_score)
{
    bid_ = CudapoaBatch::batches++;

    // Set CUDA device
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    std::string msg = " Initializing batch on device ";
    print_batch_debug_message(msg);

    // Allocate host memory and CUDA memory based on max sequence and target counts.

    // Verify that maximum sequence size is in multiples of tb size.
    // We subtract one because the matrix dimension needs to be one element larger
    // than the sequence size.
    if (CUDAPOA_MAX_SEQUENCE_SIZE % CUDAPOA_THREADS_PER_BLOCK != 0)
    {
        GW_LOG_CRITICAL("Thread block size needs to be in multiples of 32.");
        exit(-1);
    }

    initialize_input_details();

    uint32_t input_size = max_poas_ * max_sequences_per_poa_ * CUDAPOA_MAX_SEQUENCE_SIZE; //TODO how big does this need to be
    msg = " Allocated input buffers of size " + std::to_string( (static_cast<float>(input_size)  / (1024 * 1024)) ) + "MB on device ";
    print_batch_debug_message(msg);

    initialize_output_details();

    input_size += input_size * sizeof(uint16_t);
    msg = " Allocated output buffers of size " + std::to_string( (static_cast<float>(input_size)  / (1024 * 1024)) ) + "MB on device ";
    print_batch_debug_message(msg);

    initialize_alignment_details();

    initialize_graph_details();

    // Debug print for size allocated.
    uint32_t temp_size = (sizeof(int16_t) * CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION * max_poas_ );
    temp_size += 2 * (sizeof(int16_t) * CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION * max_poas_ );
    msg = " Allocated temp buffers of size " + std::to_string( (static_cast<float>(temp_size)  / (1024 * 1024)) ) + "MB on device ";
    print_batch_debug_message(msg);

    // Debug print for size allocated.
    temp_size = sizeof(uint8_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_ALIGNMENTS * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(int16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(int16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(int8_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(bool) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    temp_size += sizeof(uint16_t) * CUDAPOA_MAX_NODES_PER_WINDOW * max_poas_ ;
    msg = " Allocated temp buffers of size " + std::to_string( (static_cast<float>(temp_size)  / (1024 * 1024)) ) + "MB on device ";
    print_batch_debug_message(msg);
}

CudapoaBatch::~CudapoaBatch()
{
    std::string msg = "Destroyed buffers on device ";
    print_batch_debug_message(msg);

    free_alignment_details();
    free_graph_details();
    free_output_details();
    free_input_details();

}

uint32_t CudapoaBatch::batch_id() const
{
    return bid_;
}

uint32_t CudapoaBatch::get_total_poas() const
{
    return poa_count_;
}

void CudapoaBatch::generate_poa()
{
    GW_CU_CHECK_ERR(cudaSetDevice(device_id_));
    //Copy sequencecs, sequence lengths and window details to device
    GW_CU_CHECK_ERR(cudaMemcpyAsync(input_details_d_->sequences, input_details_h_->sequences,
                                 num_nucleotides_copied_ * sizeof(uint8_t), cudaMemcpyHostToDevice, stream_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(input_details_d_->window_details, input_details_h_->window_details,
                                 poa_count_ * sizeof(genomeworks::cudapoa::WindowDetails), cudaMemcpyHostToDevice, stream_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(input_details_d_->sequence_lengths, input_details_h_->sequence_lengths,
                                 global_sequence_idx_ * sizeof(uint16_t), cudaMemcpyHostToDevice, stream_));

    // Launch kernel to run 1 POA per thread in thread block.
    std::string msg = " Launching kernel for " + std::to_string(poa_count_) + " on device ";
    print_batch_debug_message(msg);

    genomeworks::cudapoa::generatePOA(output_details_d_,
                                 input_details_d_,
                                 poa_count_,
                                 stream_,
                                 alignment_details_d_,
                                 graph_details_d_,
                                 gap_score_,
                                 mismatch_score_,
                                 match_score_);
 
    GW_CU_CHECK_ERR(cudaPeekAtLastError());
    msg = " Launched kernel on device ";
    print_batch_debug_message(msg);
}

void CudapoaBatch::get_consensus(std::vector<std::string>& consensus,
        std::vector<std::vector<uint16_t>>& coverage)
{
    std::string msg = " Launching memcpy D2H on device ";
    print_batch_debug_message(msg);
    GW_CU_CHECK_ERR(cudaMemcpyAsync(output_details_h_->consensus,
				   output_details_d_->consensus,
				   CUDAPOA_MAX_SEQUENCE_SIZE * max_poas_ * sizeof(uint8_t),
				   cudaMemcpyDeviceToHost,
				   stream_));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(output_details_h_->coverage,
				   output_details_d_->coverage,
				   CUDAPOA_MAX_SEQUENCE_SIZE * max_poas_ * sizeof(uint16_t),
				   cudaMemcpyDeviceToHost,
				   stream_));
    GW_CU_CHECK_ERR(cudaStreamSynchronize(stream_));

    msg = " Finished memcpy D2H on device ";
    print_batch_debug_message(msg);

    for(uint32_t poa = 0; poa < poa_count_; poa++)
    {
        // Get the consensus string and reverse it since on GPU the
        // string is built backwards..
        char* c = reinterpret_cast<char *>(&(output_details_h_->consensus[poa * CUDAPOA_MAX_SEQUENCE_SIZE]));
        consensus.emplace_back(std::string(c));
        std::reverse(consensus.back().begin(), consensus.back().end());

        // Similarly, get the coverage and reverse it.
        coverage.emplace_back(std::vector<uint16_t>(
            &(output_details_h_->coverage[poa * CUDAPOA_MAX_SEQUENCE_SIZE]),
            &(output_details_h_->coverage[poa * CUDAPOA_MAX_SEQUENCE_SIZE + consensus.back().size()])));
        std::reverse(coverage.back().begin(), coverage.back().end());

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
    window_details.seq_len_buffer_offset = global_sequence_idx_;
    window_details.seq_starts = num_nucleotides_copied_;
    input_details_h_->window_details[poa_count_] = window_details;
    poa_count_++;

    return StatusType::success;
}

void CudapoaBatch::reset()
{
    poa_count_ = 0;
    num_nucleotides_copied_ = 0;
    global_sequence_idx_ = 0;
}

StatusType CudapoaBatch::add_seq_to_poa(const char* seq, uint32_t seq_len)
{
    if (seq_len >= CUDAPOA_MAX_SEQUENCE_SIZE)
    {
        return StatusType::exceeded_maximum_sequence_size;
    }

    WindowDetails *window_details = &(input_details_h_->window_details[poa_count_ - 1]);
    
    if (static_cast<uint32_t>(window_details->num_seqs) + 1 >= max_sequences_per_poa_)
    {
        return StatusType::exceeded_maximum_sequences_per_poa;
    }

    window_details->num_seqs++;
    memcpy(&(input_details_h_->sequences[num_nucleotides_copied_]),
           seq,
           seq_len);
    input_details_h_->sequence_lengths[global_sequence_idx_] = seq_len;

    num_nucleotides_copied_ += seq_len;
    global_sequence_idx_++;

    return StatusType::success;
}

} // namespace cudapoa

} // namespace genomeworks
