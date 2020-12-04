/*
* Copyright 2020 NVIDIA CORPORATION.
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
#include <cudaextender_file_location.hpp>
#include <claraparabricks/genomeworks/cudaextender/extender.hpp>
#include <claraparabricks/genomeworks/cudaextender/utils.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>
#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cuda.h>
#include <getopt.h>

using namespace claraparabricks::genomeworks;
using namespace cudautils;
using namespace cudaextender;

static void print_scored_segment_pairs(const std::vector<ScoredSegmentPair>& scored_segment_pairs)
{
    std::cout << "Target Position, Query Position, Length, Score" << std::endl;
    for (const auto& segment : scored_segment_pairs)
    {
        std::cout << segment.seed_pair.target_position_in_read << "," << segment.seed_pair.query_position_in_read
                  << "," << segment.length << "," << segment.score << std::endl;
    }
}

int main(int argc, char* argv[])
{
    const int32_t xdrop_threshold = 910;
    const bool input_no_entropy   = false;
    const int32_t score_threshold = 3000;
    char c;
    bool print               = false;
    bool help                = false;
    bool device_ptr_api_mode = false;
    while ((c = getopt(argc, argv, "pdh")) != -1)
    {
        switch (c)
        {
        case 'p':
            print = true;
            break;
        case 'd':
            device_ptr_api_mode = true;
            break;
        case 'h':
        default:
            help = true;
            break;
        }
    }

    if (help)
    {
        std::cout << "CUDAExtender API sample program. Runs ungapped extender on canned data." << std::endl;
        std::cout << "-p : Print the Scored Segment Pair output to stdout." << std::endl;
        std::cout << "-d : Use Device Pointer API. If not provided uses Host Pointer API." << std::endl;
        std::cout << "-h : Print help message." << std::endl;
        std::exit(0);
    }

    // Fasta query and target files
    const std::string target_file_path                   = std::string(CUDAEXTENDER_DATA_DIR) + "/sample.fa";
    std::unique_ptr<io::FastaParser> fasta_parser_target = io::create_kseq_fasta_parser(target_file_path, 0, false);
    // Assumes that only one sequence is present per file
    const std::string target_sequence = fasta_parser_target->get_sequence_by_id(0).seq;

    const std::string query_file_path                   = std::string(CUDAEXTENDER_DATA_DIR) + "/sample.fa";
    std::unique_ptr<io::FastaParser> fasta_parser_query = io::create_kseq_fasta_parser(query_file_path, 0, false);
    // Assumes that only one sequence is present per file
    const std::string query_sequence = fasta_parser_query->get_sequence_by_id(0).seq;

    // CSV SeedPairs file - Each row -> query_position_in_read_,
    // target_position_in_read_
    const std::string seed_pairs_file_path = std::string(CUDAEXTENDER_DATA_DIR) + "/sample_seed_pairs.csv";

    std::vector<SeedPair> h_seed_pairs;
    // Following function loops through all seed_pairs in the sample_seed_pairs.csv and returns
    // results in the passed vector
    parse_seed_pairs(seed_pairs_file_path, h_seed_pairs);
    std::cerr << "Number of Seed Pairs: " << get_size(h_seed_pairs) << std::endl;

    // Define Scoring Matrix
    const int32_t score_matrix[NUC2] = {91, -114, -31, -123, -1000, -1000, -100, -9100,
                                        -114, 100, -125, -31, -1000, -1000, -100, -9100,
                                        -31, -125, 100, -114, -1000, -1000, -100, -9100,
                                        -123, -31, -114, 91, -1000, -1000, -100, -9100,
                                        -1000, -1000, -1000, -1000, -1000, -1000, -1000, -9100,
                                        -1000, -1000, -1000, -1000, -1000, -1000, -1000, -9100,
                                        -100, -100, -100, -100, -1000, -1000, -100, -9100,
                                        -9100, -9100, -9100, -9100, -9100, -9100, -9100, -9100};

    // Allocate pinned memory for query and target strings
    pinned_host_vector<int8_t> h_encoded_target(get_size(target_sequence));
    pinned_host_vector<int8_t> h_encoded_query(get_size(query_sequence));

    encode_sequence(h_encoded_target.data(), target_sequence.c_str(), get_size(target_sequence));
    encode_sequence(h_encoded_query.data(), query_sequence.c_str(), get_size(query_sequence));
    // Create a stream for async use
    CudaStream stream0 = make_cuda_stream();
    // Create an allocator for use with both APIs
    const int64_t max_gpu_memory     = cudautils::find_largest_contiguous_device_memory_section();
    DefaultDeviceAllocator allocator = create_default_device_allocator(max_gpu_memory);
    // Reference for output
    std::vector<ScoredSegmentPair> h_ssp;
    // Create Ungapped Extender Object for both API modes
    std::unique_ptr<Extender> ungapped_extender = create_extender(score_matrix,
                                                                  NUC2,
                                                                  xdrop_threshold,
                                                                  input_no_entropy,
                                                                  stream0.get(),
                                                                  0,
                                                                  allocator);
    if (!device_ptr_api_mode)
    {
        ungapped_extender->extend_async(h_encoded_query.data(),
                                        get_size(h_encoded_query),
                                        h_encoded_target.data(),
                                        get_size(h_encoded_target),
                                        score_threshold,
                                        h_seed_pairs);
        ungapped_extender->sync();
        h_ssp = ungapped_extender->get_scored_segment_pairs();
    }
    else
    {
        // Allocate space on device for target and query sequences, seed_pairs,
        // scored segment pairs (ssp) and num_ssp using default allocator (caching)
        // Allocate space for query and target sequences
        device_buffer<int8_t> d_query(get_size(query_sequence), allocator, stream0.get());
        device_buffer<int8_t> d_target(get_size(target_sequence), allocator, stream0.get());
        // Allocate space for SeedPair input
        device_buffer<SeedPair> d_seed_pairs(get_size(h_seed_pairs), allocator, stream0.get());
        // Allocate space for ScoredSegmentPair output
        device_buffer<ScoredSegmentPair> d_ssp(get_size(h_seed_pairs), allocator, stream0.get());
        device_buffer<int32_t> d_num_ssp(1, allocator, stream0.get());

        // Async Memcopy all the input values to device
        device_copy_n(h_encoded_query.data(), get_size(query_sequence), d_query.data(), stream0.get());
        device_copy_n(h_encoded_target.data(), get_size(target_sequence), d_target.data(), stream0.get());
        device_copy_n(h_seed_pairs.data(), get_size(h_seed_pairs), d_seed_pairs.data(), stream0.get());

        // Launch the ungapped extender device pointer function
        ungapped_extender->extend_async(d_query.data(),
                                        get_size(d_query),
                                        d_target.data(),
                                        get_size(d_target),
                                        score_threshold,
                                        d_seed_pairs.data(),
                                        get_size(d_seed_pairs),
                                        d_ssp.data(),
                                        d_num_ssp.data());

        // Wait for ungapped extender to finish
        GW_CU_CHECK_ERR(cudaStreamSynchronize(stream0.get()));
        const int32_t h_num_ssp = cudautils::get_value_from_device(d_num_ssp.data(), stream0.get());
        h_ssp.resize(h_num_ssp);
        // Copy data asynchronously
        device_copy_n(d_ssp.data(), h_num_ssp, h_ssp.data(), stream0.get());
        cudaStreamSynchronize(stream0.get());
    }
    std::cerr << "Number of Scored Segment Pairs found: " << get_size(h_ssp) << std::endl;
    if (print)
    {
        print_scored_segment_pairs(h_ssp);
    }
    return 0;
}
