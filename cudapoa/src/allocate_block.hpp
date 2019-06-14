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

#include <memory>
#include <vector>
#include <stdint.h>
#include <string>

#include <cuda_runtime_api.h>

#include "cudapoa_kernels.cuh"

namespace genomeworks
{

namespace cudapoa
{

class BatchBlock
{
public:
    BatchBlock(int32_t device_id_, int32_t max_poas, int32_t max_sequences_per_poa, bool banded_alignment = false);
    ~BatchBlock();

    void get_output_details(OutputDetails** output_details_h_p, OutputDetails** output_details_d_p);

    void get_input_details(InputDetails** input_details_h_p, InputDetails** input_details_d_p);

    void get_alignment_details(AlignmentDetails** alignment_details_d_p);

    void get_graph_details(GraphDetails** graph_details_d_p);

    uint8_t* get_block_host();

    uint8_t* get_block_device();

protected:
    void calculate_size();

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
    uint64_t total_h_ = 0;
    uint64_t total_d_ = 0;

    // Offset index for pointing a buffer to block memory
    uint64_t offset_h_ = 0;
    uint64_t offset_d_ = 0;

    int32_t input_size_;
    int32_t output_size_;
    int32_t matrix_sequence_dimension_;
    int32_t max_graph_dimension_;
    uint16_t max_nodes_per_window_;
    int32_t device_id_;
};

} // namespace cudapoa

} // namespace genomeworks
