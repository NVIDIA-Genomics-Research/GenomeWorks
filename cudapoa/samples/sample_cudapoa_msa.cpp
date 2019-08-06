/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../benchmarks/common/utils.hpp"

#include <file_location.hpp>
#include <claragenomics/cudapoa/cudapoa.hpp>
#include <claragenomics/cudapoa/batch.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/utils/cudautils.hpp>

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <stdexcept>

int main()
{
    // Load input data. Each POA group is represented as a vector of strings. The sample
    // data has many such POA groups to process, hence the data is loaded into a vector
    // of cvector of string.
    const std::string input_data = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt";
    std::vector<std::vector<std::string>> windows;
    claragenomics::cudapoa::parse_window_data_file(windows, input_data, 100); // Generate 100 windows.
    assert(claragenomics::get_size(windows) > 0);

    // Get device information.
    int32_t device_count = 0;
    CGA_CU_CHECK_ERR(cudaGetDeviceCount(&device_count));
    assert(device_count > 0);

    size_t total = 0, free = 0;
    cudaSetDevice(0); // Using first GPU for sample.
    cudaMemGetInfo(&free, &total);

    // Initialize internal logging framework.
    claragenomics::cudapoa::Init();

    // Initialize CUDAPOA batch object for batched processing of POAs on the GPU.
    const int32_t max_sequences_per_poa_group = 500;
    const int32_t device_id                   = 0;
    cudaStream_t stream                       = 0;
    size_t mem_per_batch                      = 0.9 * free; // Using 90% of GPU available memory for CUDAPOA batch.
    const int32_t mismatch_score = -6, gap_score = -8, match_score = 8;
    bool banded_alignment = false;

    std::unique_ptr<claragenomics::cudapoa::Batch> batch = claragenomics::cudapoa::create_batch(max_sequences_per_poa_group,
                                                                                                device_id,
                                                                                                stream,
                                                                                                mem_per_batch,
                                                                                                claragenomics::cudapoa::OutputType::msa,
                                                                                                gap_score,
                                                                                                mismatch_score,
                                                                                                match_score,
                                                                                                banded_alignment);

    // Loop over all the POA groups, add them to the batch and process them.
    for (int32_t i = 0; i < claragenomics::get_size(windows);)
    {
        const std::vector<std::string>& window = windows[i];

        claragenomics::cudapoa::Group poa_group;
        // Create a new entry for each sequence and add to the group.
        for (const auto& seq : window)
        {
            claragenomics::cudapoa::Entry poa_entry{};
            poa_entry.seq     = seq.c_str();
            poa_entry.length  = seq.length();
            poa_entry.weights = nullptr;
            poa_group.push_back(poa_entry);
        }

        std::vector<claragenomics::cudapoa::StatusType> seq_status;
        claragenomics::cudapoa::StatusType status = batch->add_poa_group(seq_status, poa_group);

        // NOTE: If number of windows smaller than batch capacity, then run POA generation
        // once last window is added to batch.
        if (status == claragenomics::cudapoa::StatusType::exceeded_maximum_poas || (i == claragenomics::get_size(windows) - 1))
        {
            // No more POA groups can be added to batch. Now process batch.
            batch->generate_poa();

            // Grab MSA results for all POA groups in batch.
            std::vector<std::vector<std::string>> msa;                     // MSA per group
            std::vector<claragenomics::cudapoa::StatusType> output_status; // Status of MSA generation per group

            status = batch->get_msa(msa, output_status);
            if (status != claragenomics::cudapoa::StatusType::success)
            {
                std::cerr << "Could not generate MSA for batch : " << status << std::endl;
            }

            for (int32_t g = 0; g < claragenomics::get_size(msa); g++)
            {
                if (output_status[g] != claragenomics::cudapoa::StatusType::success)
                {
                    std::cerr << "Error generating  MSA for POA group " << g << ". Error type " << output_status[g] << std::endl;
                }
                else
                {
                    for (const auto& alignment : msa[g])
                    {
                        std::cout << alignment << std::endl;
                    }
                }
            }

            // After MSA is generated for batch, reset batch to make roomf or next set of POA groups.
            batch->reset();

            // Increment counter to keep track of loop in case windows < batch capacity
            if (i < claragenomics::get_size(windows))
            {
                i++;
            }
        }
        else if (status == claragenomics::cudapoa::StatusType::success)
        {
            // Check if all sequences in POA group wre added successfully.
            for (const auto& s : seq_status)
            {
                if (s == claragenomics::cudapoa::StatusType::exceeded_maximum_sequence_size)
                {
                    std::cout << "Dropping sequence because sequence exceeded maximum size" << std::endl;
                }
                else if (s == claragenomics::cudapoa::StatusType::exceeded_maximum_sequences_per_poa)
                {
                    std::cout << "Dropping sequence because maximum sequences per POA group exceeded" << std::endl;
                }
            }
            i++;
        }
        else
        {
            std::cerr << "Could not add POA group to batch. Error code " << status << std::endl;
        }
    }

    return 0;
}
