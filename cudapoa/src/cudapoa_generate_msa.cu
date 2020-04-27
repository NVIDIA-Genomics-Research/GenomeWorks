/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "cudapoa_kernels.cuh"

#include <claragenomics/utils/cudautils.hpp>

#include <stdio.h>

namespace claragenomics
{

namespace cudapoa
{

template <typename SizeT>
__device__ SizeT getNodeIDToMSAPosDevice(SizeT node_count,
                                         SizeT* sorted_poa,
                                         SizeT* node_id_to_msa_pos,
                                         uint16_t* node_alignment_counts)
{
    SizeT msa_pos = 0;
    for (SizeT rank = 0; rank < node_count; rank++)
    {
        SizeT node_id               = sorted_poa[rank];
        node_id_to_msa_pos[node_id] = msa_pos;
        uint16_t alignment_count    = node_alignment_counts[node_id];
        for (uint16_t n = 0; n < alignment_count; n++)
        {
            node_id_to_msa_pos[sorted_poa[++rank]] = msa_pos;
        }
        msa_pos++;
    }
    return msa_pos;
}

template <typename SizeT>
__device__ void generateMSADevice(uint8_t* nodes,
                                  uint16_t num_sequences,
                                  uint16_t* outgoing_edge_count,
                                  SizeT* outgoing_edges,
                                  uint16_t* outgoing_edges_coverage,       //fill this pointer with seq ids in addAlignmentToGraph whenever a new edge is added
                                  uint16_t* outgoing_edges_coverage_count, //ditto
                                  SizeT* node_id_to_msa_pos,               //this is calculated with the getNodeIDToMSAPos device function above
                                  SizeT* sequence_begin_nodes_ids,         //fill this pointer in the generatePOAKernel
                                  uint8_t* multiple_sequence_alignments,
                                  uint16_t msa_length,
                                  uint32_t max_sequences_per_poa,
                                  uint32_t max_limit_consensus_size)
{
    // each thread operate on a sequence
    uint16_t s = threadIdx.x;
    if (s >= num_sequences)
        return;

    SizeT node_id      = sequence_begin_nodes_ids[s];
    SizeT filled_until = 0;
    while (true)
    {
        SizeT msa_pos                                                        = node_id_to_msa_pos[node_id];
        multiple_sequence_alignments[s * max_limit_consensus_size + msa_pos] = nodes[node_id];

        // fill the gap in current alignment with '-'
        if (msa_pos > filled_until)
        {
            for (SizeT i = filled_until; i < msa_pos; i++)
            {
                multiple_sequence_alignments[s * max_limit_consensus_size + i] = '-';
            }
        }
        filled_until = msa_pos + 1;

        //check if the current node has an outgoing edge on this sequnece, i.e. if it's the last node of the seq
        bool end_node = true;
        for (uint16_t n = 0; n < outgoing_edge_count[node_id]; n++)
        {
            SizeT to_node = outgoing_edges[node_id * CUDAPOA_MAX_NODE_EDGES + n];
            for (uint16_t m = 0; m < outgoing_edges_coverage_count[node_id * CUDAPOA_MAX_NODE_EDGES + n]; m++)
            {
                uint16_t curr_edge_seq = outgoing_edges_coverage[node_id * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa +
                                                                 n * max_sequences_per_poa + m];
                //found an out edge on the current sequence, move to the next node
                if (curr_edge_seq == s)
                {
                    end_node = false;
                    node_id  = to_node;
                    break;
                }
            }
            if (!end_node)
                break; //found the next node on the sequence
        }
        // if reach the end of alignment, filled the remaining spaces with '-'
        if (end_node)
        {
            if (filled_until < msa_length)
            {
                for (SizeT i = filled_until; i < msa_length; i++)
                {
                    multiple_sequence_alignments[s * max_limit_consensus_size + i] = '-';
                }
            }
            break; //no next node, break out of the while loop and we have found a complete sequence.
        }
    }
    multiple_sequence_alignments[s * max_limit_consensus_size + msa_length] = '\0';
}

template <bool cuda_banded_alignment = false, typename SizeT>
__global__ void generateMSAKernel(uint8_t* nodes_d,
                                  uint8_t* consensus_d,
                                  claragenomics::cudapoa::WindowDetails* window_details_d,
                                  uint16_t* incoming_edge_count_d,
                                  SizeT* incoming_edges_d,
                                  uint16_t* outgoing_edge_count_d,
                                  SizeT* outgoing_edges_d,
                                  uint16_t* outgoing_edges_coverage_d, //fill this pointer in addAlignmentKernel whenever a new edge is added
                                  uint16_t* outgoing_edges_coverage_count_d,
                                  SizeT* node_id_to_msa_pos_d,       //this is calculated with the getNodeIDToMSAPos function above
                                  SizeT* sequence_begin_nodes_ids_d, //fill this pointer in the generatePOAKernel
                                  uint8_t* multiple_sequence_alignments_d,
                                  SizeT* sequence_lengths_d,
                                  SizeT* sorted_poa_d,
                                  SizeT* node_alignments_d,
                                  uint16_t* node_alignment_counts_d,
                                  uint32_t max_sequences_per_poa,
                                  SizeT* node_id_to_pos_d,
                                  uint8_t* node_marks_d,
                                  bool* check_aligned_nodes_d,
                                  SizeT* nodes_to_visit_d,
                                  uint32_t max_limit_nodes_per_window,
                                  uint32_t max_limit_nodes_per_window_banded,
                                  uint32_t max_limit_consensus_size)
{
    //each block of threads will operate on a window
    uint32_t window_idx = blockIdx.x;

    uint8_t* consensus = &consensus_d[window_idx * max_limit_consensus_size];

    if (consensus[0] == CUDAPOA_KERNEL_ERROR_ENCOUNTERED) //error during graph generation
        return;

    uint32_t max_nodes_per_window = cuda_banded_alignment ? max_limit_nodes_per_window_banded : max_limit_nodes_per_window;

    // Find the buffer offsets for each thread within the global memory buffers.
    uint8_t* nodes                          = &nodes_d[max_nodes_per_window * window_idx];
    uint16_t* outgoing_edge_count           = &outgoing_edge_count_d[window_idx * max_nodes_per_window];
    SizeT* outgoing_edges                   = &outgoing_edges_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edges_coverage       = &outgoing_edges_coverage_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES * max_sequences_per_poa];
    uint16_t* outgoing_edges_coverage_count = &outgoing_edges_coverage_count_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    SizeT* node_id_to_msa_pos               = &node_id_to_msa_pos_d[window_idx * max_nodes_per_window];
    SizeT* sequence_begin_nodes_ids         = &sequence_begin_nodes_ids_d[window_idx * max_sequences_per_poa];
    uint8_t* multiple_sequence_alignments   = &multiple_sequence_alignments_d[window_idx * max_sequences_per_poa * max_limit_consensus_size];
    SizeT* sorted_poa                       = &sorted_poa_d[window_idx * max_nodes_per_window];
    uint16_t* node_alignment_counts         = &node_alignment_counts_d[window_idx * max_nodes_per_window];
    uint32_t num_sequences                  = window_details_d[window_idx].num_seqs;
    SizeT* sequence_lengths                 = &sequence_lengths_d[window_details_d[window_idx].seq_len_buffer_offset];
    SizeT* incoming_edges                   = &incoming_edges_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* incoming_edge_count           = &incoming_edge_count_d[window_idx * max_nodes_per_window];
    SizeT* node_alignments                  = &node_alignments_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_ALIGNMENTS];
    uint8_t* node_marks                     = &node_marks_d[max_nodes_per_window * window_idx];
    bool* check_aligned_nodes               = &check_aligned_nodes_d[max_nodes_per_window * window_idx];
    SizeT* nodes_to_visit                   = &nodes_to_visit_d[max_nodes_per_window * window_idx];
    SizeT* node_id_to_pos                   = &node_id_to_pos_d[window_idx * max_nodes_per_window];

    __shared__ SizeT msa_length;

    if (threadIdx.x == 0) //one thread will fill in node_id_to_msa_pos and msa_length for it's block/window
    {
        // Exactly matches racon CPU results
        raconTopologicalSortDeviceUtil(sorted_poa,
                                       node_id_to_pos,
                                       sequence_lengths[0],
                                       incoming_edge_count,
                                       incoming_edges,
                                       node_alignment_counts,
                                       node_alignments,
                                       node_marks,
                                       check_aligned_nodes,
                                       nodes_to_visit,
                                       cuda_banded_alignment,
                                       (uint16_t)max_nodes_per_window);

        msa_length = getNodeIDToMSAPosDevice<SizeT>(sequence_lengths[0],
                                                    sorted_poa,
                                                    node_id_to_msa_pos,
                                                    node_alignment_counts);

        if (msa_length >= max_limit_consensus_size)
        {
            consensus[0] = CUDAPOA_KERNEL_ERROR_ENCOUNTERED;
            consensus[1] = static_cast<uint8_t>(StatusType::exceeded_maximum_sequence_size);
        }
    }

    __syncthreads();

    if (consensus[0] == CUDAPOA_KERNEL_ERROR_ENCOUNTERED)
        return;

    generateMSADevice<SizeT>(nodes,
                             num_sequences,
                             outgoing_edge_count,
                             outgoing_edges,
                             outgoing_edges_coverage,
                             outgoing_edges_coverage_count,
                             node_id_to_msa_pos,
                             sequence_begin_nodes_ids,
                             multiple_sequence_alignments,
                             msa_length,
                             max_sequences_per_poa,
                             max_limit_consensus_size);
}

} // namespace cudapoa

} // namespace claragenomics
