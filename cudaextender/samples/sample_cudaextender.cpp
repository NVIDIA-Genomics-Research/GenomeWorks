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
#include <file_location.hpp>
#include <claraparabricks/genomeworks/cudaextender/extender.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>
#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda.h>

using namespace claraparabricks::genomeworks;
using namespace cudautils;
using namespace cudaextender;

constexpr char A_NT = 0;
constexpr char C_NT = 1;
constexpr char G_NT = 2;
constexpr char T_NT = 3;
constexpr char L_NT = 4;
constexpr char N_NT = 5;
constexpr char X_NT = 6;
constexpr char E_NT = 7;
constexpr char NUC  = 8;
constexpr char NUC2 = NUC * NUC;

// Really simple parser with no error checks
void parse_seed_pairs(const std::string& filepath, std::vector<SeedPair>& seed_pairs)
{
    std::ifstream seed_pair_file(filepath);
    if (!seed_pair_file.is_open())
        throw std::runtime_error("Cannot open file");
    if (seed_pair_file.good())
    {
        std::string line;
        while (std::getline(seed_pair_file, line, ','))
        {
            SeedPair seed_pair;
            seed_pair.target_position_in_read = std::atoi(line.c_str());
            std::getline(seed_pair_file, line); // Get the next value
            seed_pair.query_position_in_read = std::atoi(line.c_str());
            seed_pairs.push_back(seed_pair);
        }
    }
}

// convert input sequence from alphabet to integers
void encode_string(char* dst_seq, const char* src_seq, int32_t len)
{
    for (int32_t i = 0; i < len; i++)
    {
        char ch = src_seq[i];
        char dst;
        switch (ch)
        {
        case 'A':
            dst_seq[i] = A_NT;
            break;
        case 'C':
            dst_seq[i] = C_NT;
            break;
        case 'G':
            dst_seq[i] = G_NT;
            break;
        case 'T':
            dst_seq[i] = T_NT;
            break;
        case '&':
            dst_seq[i] = E_NT;
            break;
        case 'n':
        case 'N':
            dst_seq[i] = N_NT;
            break;
        case 'a':
        case 'c':
        case 'g':
        case 't':
            dst_seq[i] = L_NT;
            break;
        default:
            dst_seq[i] = X_NT;
            break;
        }
    }
}

void print_scored_segment_pairs(std::vector<ScoredSegmentPair> scored_segment_pairs)
{
    std::cout << "Target Position, Query Position, Length, Score" << std::endl;
    for (auto& segment : scored_segment_pairs)
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
    std::string target_file_path                         = std::string(CUDAEXTENDER_DATA_DIR) + "/sample.fa";
    std::unique_ptr<io::FastaParser> fasta_parser_target = io::create_kseq_fasta_parser(target_file_path, 0, false);
    // Assumes that only one sequence is present per file
    std::string target_sequence = fasta_parser_target->get_sequence_by_id(0).seq;

    std::string query_file_path = std::string(CUDAEXTENDER_DATA_DIR) + "/sample.fa";
    ;
    std::unique_ptr<io::FastaParser> fasta_parser_query =
        io::create_kseq_fasta_parser(query_file_path, 0, false);
    // Assumes that only one sequence is present per file
    std::string query_sequence = fasta_parser_query->get_sequence_by_id(0).seq;

    // CSV SeedPairs file - Each row -> query_position_in_read_,
    // target_position_in_read_
    std::string seed_pairs_file_path = std::string(CUDAEXTENDER_DATA_DIR) + "/sample_seed_pairs.csv";

    //TODO - pinned seed_pairs
    std::vector<SeedPair> h_seed_pairs;
    // Following function loops through all seed_pairs in the sample_seed_pairs.csv and returns
    // results in
    // the passed vector
    parse_seed_pairs(seed_pairs_file_path, h_seed_pairs);
    std::cerr << "Number of seed pairs: " << h_seed_pairs.size() << std::endl;

    // Define Scoring Matrix
    int32_t score_matrix[NUC2] = {91, -114, -31, -123, -1000, -1000, -100, -9100,
                                  -114, 100, -125, -31, -1000, -1000, -100, -9100,
                                  -31, -125, 100, -114, -1000, -1000, -100, -9100,
                                  -123, -31, -114, 91, -1000, -1000, -100, -9100,
                                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -9100,
                                  -1000, -1000, -1000, -1000, -1000, -1000, -1000, -9100,
                                  -100, -100, -100, -100, -1000, -1000, -100, -9100,
                                  -9100, -9100, -9100, -9100, -9100, -9100, -9100, -9100};

    // Allocate pinned memory for query and target strings
    pinned_host_vector<char> h_encoded_target(target_sequence.length());
    pinned_host_vector<char> h_encoded_query(target_sequence.length());

    encode_string(h_encoded_target.data(), target_sequence.c_str(), target_sequence.length());
    encode_string(h_encoded_query.data(), query_sequence.c_str(), query_sequence.length());
    // Create a stream for async use
    CudaStream stream0 = make_cuda_stream();
    // Create an allocator for use with both APIs
    const std::size_t max_gpu_memory = cudautils::find_largest_contiguous_device_memory_section();
    DefaultDeviceAllocator allocator = create_default_device_allocator(max_gpu_memory);

    if (!device_ptr_api_mode)
    {
        std::unique_ptr<Extender> ungapped_extender = create_extender(score_matrix, NUC2, xdrop_threshold, input_no_entropy, stream0.get(), 0, allocator);
        ungapped_extender->extend_async(h_encoded_query.data(), h_encoded_query.size(), h_encoded_target.data(), h_encoded_target.size(), score_threshold, h_seed_pairs);
        ungapped_extender->sync();
        std::vector<ScoredSegmentPair> h_ssp = ungapped_extender->get_scored_segment_pairs();
        std::cerr << "Number of ScoredSegmentPairs found: " << h_ssp.size() << std::endl;
        if (print)
            print_scored_segment_pairs(h_ssp);
    }
    else
    {
        // Allocate space on device for target and query sequences, seed_pairs,
        // scored segment pairs (ssp) and num_ssp using default allocator (caching)
        // Allocate space for query and target sequences
        device_buffer<char> d_query(query_sequence.length(), allocator, stream0.get());
        device_buffer<char> d_target(target_sequence.length(), allocator, stream0.get());
        // Allocate space for SeedPair input
        device_buffer<SeedPair> d_seed_pairs(h_seed_pairs.size(), allocator, stream0.get());
        // Allocate space for ScoredSegmentPair output
        device_buffer<ScoredSegmentPair> d_ssp(h_seed_pairs.size(), allocator, stream0.get());
        // TODO - Keep this as a malloc for single int?
        int32_t* d_num_ssp;
        GW_CU_CHECK_ERR(cudaMalloc((void**)&d_num_ssp, sizeof(int32_t)));

        // Async Memcopy all the input values to device
        device_copy_n(h_encoded_query.data(), query_sequence.length(), d_query.data(), stream0.get());
        device_copy_n(h_encoded_target.data(), target_sequence.length(), d_target.data(), stream0.get());
        device_copy_n(h_seed_pairs.data(), h_seed_pairs.size(), d_seed_pairs.data(), stream0.get());

        // Create an ungapped extender object
        std::unique_ptr<Extender> ungapped_extender = create_extender(score_matrix, NUC2, xdrop_threshold, input_no_entropy, stream0.get(), 0, allocator);

        // Launch the ungapped extender device function
        ungapped_extender->extend_async(d_query.data(), // Type TBD based on encoding
                                        d_query.size(),
                                        d_target.data(),
                                        d_target.size(),
                                        score_threshold,
                                        d_seed_pairs.data(),
                                        d_seed_pairs.size(),
                                        d_ssp.data(),
                                        d_num_ssp);

        // Wait for ungapped extender to finish
        GW_CU_CHECK_ERR(cudaStreamSynchronize(stream0.get()));
        int32_t h_num_ssp = cudautils::get_value_from_device(d_num_ssp, stream0.get());
        //Get results
        std::cerr << "Number of ScoredSegmentPairs found: " << h_num_ssp << std::endl;
        std::vector<ScoredSegmentPair> h_ssp(h_num_ssp);
        // Copy data synchronously
        device_copy_n(d_ssp.data(), h_num_ssp, h_ssp.data());
        if (print)
            print_scored_segment_pairs(h_ssp);
        // Free all CUDA allocated memory
        GW_CU_CHECK_ERR(cudaFree(d_num_ssp));
    }

    return 0;
}
