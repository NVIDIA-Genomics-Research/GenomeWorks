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

#include <claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp>
#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>
#include <claraparabricks/genomeworks/cudaaligner/alignment.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/genomeutils.hpp>

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <random>

using namespace claraparabricks::genomeworks;
using namespace claraparabricks::genomeworks::genomeutils;
using namespace claraparabricks::genomeworks::cudaaligner;

std::unique_ptr<Aligner> initialize_batch(int32_t max_query_size,
                                          int32_t max_target_size,
                                          int32_t max_alignments_per_batch,
                                          DefaultDeviceAllocator allocator)
{
    // Get device information.
    int32_t device_count = 0;
    GW_CU_CHECK_ERR(cudaGetDeviceCount(&device_count));
    assert(device_count > 0);

    // Initialize internal logging framework.
    Init();

    // Initialize CUDA Aligner batch object for batched processing of alignments on the GPU.
    const int32_t device_id = 0;
    cudaStream_t stream     = 0;

    std::unique_ptr<Aligner> batch = create_aligner(max_query_size,
                                                    max_target_size,
                                                    max_alignments_per_batch,
                                                    AlignmentType::global_alignment,
                                                    allocator,
                                                    stream,
                                                    device_id);

    return std::move(batch);
}

void generate_data(std::vector<std::pair<std::string, std::string>>& data,
                   int32_t max_query_size,
                   int32_t max_target_size,
                   int32_t num_examples)
{
    std::minstd_rand rng(1);
    for (int32_t i = 0; i < num_examples; i++)
    {
        data.emplace_back(std::make_pair(
            generate_random_genome(max_query_size, rng),
            generate_random_genome(max_target_size, rng)));
    }
}

int main(int argc, char** argv)
{
    // Process options
    int c      = 0;
    bool help  = false;
    bool print = false;

    while ((c = getopt(argc, argv, "hp")) != -1)
    {
        switch (c)
        {
        case 'p':
            print = true;
            break;
        case 'h':
            help = true;
            break;
        }
    }

    if (help)
    {
        std::cout << "CUDA Aligner API sample program. Runs pairwise alignment over a batch of randomly generated sequences." << std::endl;
        std::cout << "Usage:" << std::endl;
        std::cout << "./sample_cudaaligner [-p] [-h]" << std::endl;
        std::cout << "-p : Print the MSA or consensus output to stdout" << std::endl;
        std::cout << "-h : Print help message" << std::endl;
        std::exit(0);
    }

    const int32_t query_length  = 10000;
    const int32_t target_length = 15000;
    const uint32_t num_entries  = 1000;

    const std::size_t max_gpu_memory = cudautils::find_largest_contiguous_device_memory_section();
    DefaultDeviceAllocator allocator = create_default_device_allocator(max_gpu_memory);

    std::cout << "Running pairwise alignment for " << num_entries << " pairs..." << std::endl;

    // Initialize batch.
    std::unique_ptr<Aligner> batch = initialize_batch(query_length, target_length, 100, allocator);

    // Generate data.
    std::vector<std::pair<std::string, std::string>> data;
    generate_data(data, query_length, target_length, num_entries);
    assert(data.size() == num_entries);

    // Loop over all the alignment pairs, add them to the batch and process them.
    uint32_t data_id = 0;
    while (data_id != num_entries)
    {
        const std::string& query  = data[data_id].first;
        const std::string& target = data[data_id].second;

        // Add a pair to the batch, and check for status.
        StatusType status = batch->add_alignment(query.c_str(), query.length(), target.c_str(), target.length());
        if (status == exceeded_max_alignments || data_id == num_entries - 1)
        {
            // Launch alignment on the GPU. align_all is an async call.
            batch->align_all();
            // Synchronize all alignments.
            batch->sync_alignments();
            if (print)
            {
                const std::vector<std::shared_ptr<Alignment>>& alignments = batch->get_alignments();
                for (const auto& alignment : alignments)
                {
                    FormattedAlignment formatted = alignment->format_alignment();
                    std::cout << formatted;
                }
            }
            // Reset batch to reuse memory for new alignments.
            batch->reset();
            std::cout << "Aligned till " << (data_id - 1) << "." << std::endl;
        }
        else if (status != success)
        {
            throw std::runtime_error("Experienced error type " + std::to_string(status));
        }

        // If alignment was add successfully, increment counter.
        if (status == success)
        {
            data_id++;
        }
    }

    return 0;
}
