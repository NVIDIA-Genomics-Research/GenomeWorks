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
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <vector>

#include <claraparabricks/genomeworks/cudaungappedextender/cudaungappedextender.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>

using namespace claraparabricks::genomeworks;
using namespace claraparabricks::genomeworks::cudaungappedextender;

int main(int argc, char* argv[])
{
    const int32_t input_xdrop = 910;
    const bool input_no_entropy = false;
    const int32_t score_threshold = 3000;
    // Fasta query and target files
    std::string target_file_path = "../data/example.fa";
    std::unique_ptr<io::FastaParser> fasta_parser_target =
        io::create_kseq_fasta_parser(target_file_path, 0, false);
    // Assumes that only one sequence is present per file
    std::string target_sequence = fasta_parser_target->get_sequence_by_id(0);

    std::string query_file_path = "../data/example.fa";
    std::unique_ptr<io::FastaParser> fasta_parser_query =
        io::create_kseq_fasta_parser(query_file_path, 0, false);
    // Assumes that only one sequence is present per file
    magic_sequence query_sequence = fasta_parser_query->get_sequence_by_id(0);

    // CSV SeedPairs file - Each row -> query_position_in_read_,
    // target_position_in_read_
    std::string seed_pairs_file_path = "../data/example_seed_pairs.csv";

    std::vector<SeedPair> h_seed_pairs;
    // Following function loops through all seed_pairs in the example_seed_pairs.csv and returns
    // results in
    // the passed vector
    parse_seed_pairs(seed_pairs_file_path, h_seed_pairs);

    // Following sections TBD based on encoding
    ScoreMatrix                  = magic_number_matrix;
    std::string h_encoded_target = magic_encode(magic_base, target_sequence);
    std::string h_encoded_query  = magic_encode(magic_base, query_sequence);

    // Create a stream for async use
    CudaStream stream0 = make_cuda_stream();
    // Allocate space on device for target and query sequences, seed_pairs,
    // high scoring segment pairs (hsps) and num_hsps.
    char* d_query;
    GW_CU_CHECK_ERROR(
        cudaMalloc((void**)&d_query, sizeof(char) * h_encoded_query.size()));
    char* d_target;
    GW_CU_CHECK_ERROR(
        cudaMalloc((void**)&d_target, sizeof(char) * h_target_query.size()));
    SeedPair* d_seed_pairs;
    GW_CU_CHECK_ERROR(
        cudaMalloc((void**)&d_seed_pairs, sizeof(SeedPair) * h_seed_pairs.size()));
    // Allocate a minimum of num_seed_pairs as all seed_pairs could be hsps in the worst case
    int32_t h_num_hsps = 0;
    ScoredSegmentPair* d_hsps;
    GW_CU_CHECK_ERROR(
        cudaMalloc((void**)&d_hsps, sizeof(ScoredSegmentPair) * h_seed_pairs.size()));
    int32_t* d_num_hsps;
    GW_CU_CHECK_ERROR(cudaMalloc((void**)&d_num_hsps, sizeof(int32_t));

                      // Async Memcopy all the input values to device
                      GW_CU_CHECK_ERR(cudaMemcpyAsync(d_query, h_encoded_query.c_str(), sizeof(char) * h_encoded_query.size(),
                                                      cudaMemcpyHostToDevice, stream0.get()));
                      GW_CU_CHECK_ERR(cudaMemcpyAsync(d_target, h_encoded_target.c_str(), sizeof(char) * h_encoded_target.size(),
                                                      cudaMemcpyHostToDevice, stream0.get()));
                      GW_CU_CHECK_ERR(cudaMemcpyAsync(d_seed_pairs, &h_seed_pairs[0], sizeof(SeedPair) * h_seed_pairs.size(), cudaMemcpyHostToDevice,
                                                      stream0.get())));

    // Create an ungapped extender object
    std::unique_ptr<UngappedExtender> ungapped_extender = std::make_unique<UngappedExtender>(0,
                                                                                             magic_number_matrix,
                                                                                             input_xdrop,
                                                                                             input_no_entropy,
                                                                                             stream0.get());
    // Launch the ungapped extender device function
    ungapped_extender->extend_async(d_query, // Type TBD based on encoding
                                    encoded_query.size(),
                                    d_target.c_str(),
                                    encoded_target.size(),
                                    score_threshold,
                                    d_seed_pairs,
                                    h_seed_pairs.size(),
                                    d_hsps,
                                    d_num_hsps);
    // Copy back the number of hsps to host
    GW_CU_CHECK_ERR(cudaMemcpyAsync(&h_num_hsps, d_num_hsps, sizeof(int32_t), cudaMemcpyDeviceToHost, stream0.get()));

    // Wait for ungapped extender to finish
    GW_CU_CHECK_ERR(cudaStreamSynchronize(stream0.get()));

    //Get results
    if (h_num_hsps > 0)
    {
        std::vector<ScoredSegmentPair> h_hsps(h_num_hsps);
        // Don't care about asynchronous copies here
        GW_CU_CHECK_ERR(cudaMemcpy(&h_hsps[0], d_hsps,
                                   sizeof(ScoredSegmentPair) * h_num_hsps,
                                   cudaMemcpyDeviceToHost));

        int32_t i = 0;
        for (const auto& segment : h_hsps)
        {
            std::cout << "Segment: " << i << "Length: " << segment.length
                      << "Score: " << segment.score << std::endl;
            std::cout << "Position in query: "
                      << segment.seed_pair.query_position_in_read << std::endl;
            std::cout << "Position in target: "
                      << segment.seed_pair.target_position_in_read << std::endl;
            i++;
        }
    }

    // Free all allocated memory on the GPU
    GW_CU_CHECK_ERROR(cudaFree(d_query);
    GW_CU_CHECK_ERROR(cudaFree(d_target);
    GW_CU_CHECK_ERROR(cudaFree(d_hsps);
    GW_CU_CHECK_ERROR(cudaFree(d_seed_pairs);
    GW_CU_CHECK_ERROR(cudaFree(d_num_hsps);

    return 0;
}
