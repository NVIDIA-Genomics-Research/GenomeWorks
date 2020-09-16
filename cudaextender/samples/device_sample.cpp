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
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cuda.h>

#include <claraparabricks/genomeworks/cudaextender/extender.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>

using namespace claraparabricks::genomeworks;
using namespace claraparabricks::genomeworks::cudaextender;

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
        char ch  = src_seq[i];
        char dst = X_NT;
        if (ch == 'A')
            dst = A_NT;
        else if (ch == 'C')
            dst = C_NT;
        else if (ch == 'G')
            dst = G_NT;
        else if (ch == 'T')
            dst = T_NT;
        else if ((ch == 'a') || (ch == 'c') || (ch == 'g') || (ch == 't'))
            dst = L_NT;
        else if ((ch == 'n') || (ch == 'N'))
            dst = N_NT;
        else if (ch == '&')
            dst = E_NT;
        dst_seq[i] = dst;
    }
}

int main(int argc, char* argv[])
{
    const int32_t xdrop_threshold = 910;
    const bool input_no_entropy   = false;
    const int32_t score_threshold = 3000;
    // Fasta query and target files
    std::string target_file_path = "../data/example.fa";
    std::unique_ptr<io::FastaParser> fasta_parser_target = io::create_kseq_fasta_parser(target_file_path, 0, false);
    // Assumes that only one sequence is present per file
    std::string target_sequence = fasta_parser_target->get_sequence_by_id(0).seq;

    std::string query_file_path = "../data/example.fa";
    std::unique_ptr<io::FastaParser> fasta_parser_query =
        io::create_kseq_fasta_parser(query_file_path, 0, false);
   // Assumes that only one sequence is present per file
    std::string query_sequence = fasta_parser_query->get_sequence_by_id(0).seq;

    // CSV SeedPairs file - Each row -> query_position_in_read_,
    // target_position_in_read_
    std::string seed_pairs_file_path = "../data/example_hits.csv";

    std::vector<SeedPair> h_seed_pairs;
    // Following function loops through all seed_pairs in the example_seed_pairs.csv and returns
    // results in
    // the passed vector
    parse_seed_pairs(seed_pairs_file_path, h_seed_pairs);
    std::cout <<"Number of seed pairs: "<<h_seed_pairs.size() << std::endl;

    // Define Scoring Matrix
    int32_t score_matrix[NUC2] = {91, -114, -31, -123, -1000, -1000, -100, -9100,
                                -114, 100, -125, -31, -1000, -1000, -100, -9100,
                                -31, -125, 100, -114, -1000, -1000, -100, -9100,
                                -123, -31, -114, 91, -1000, -1000, -100, -9100,
                                -1000, -1000, -1000, -1000, -1000, -1000, -1000, -9100,
                                -1000, -1000, -1000, -1000, -1000, -1000, -1000, -9100,
                                -100, -100, -100, -100, -1000, -1000, -100, -9100,
                                -9100, -9100, -9100, -9100, -9100, -9100, -9100, -9100};

    char* h_encoded_target = (char*)malloc(sizeof(char) * target_sequence.length());
    char* h_encoded_query  = (char*)malloc(sizeof(char) * query_sequence.length());
    encode_string(h_encoded_target, target_sequence.c_str(), target_sequence.length());
    encode_string(h_encoded_query, query_sequence.c_str(), query_sequence.length());
    // Create a stream for async use
    CudaStream stream0 = make_cuda_stream();
    // Allocate space on device for target and query sequences, seed_pairs,
    // high scoring segment pairs (hsps) and num_hsps.
    char* d_query;
    GW_CU_CHECK_ERR(cudaMalloc((void**)&d_query, sizeof(char) * query_sequence.length()));
    char* d_target;
    GW_CU_CHECK_ERR(cudaMalloc((void**)&d_target, sizeof(char) * target_sequence.length()));
    SeedPair* d_seed_pairs;
    GW_CU_CHECK_ERR(cudaMalloc((void**)&d_seed_pairs, sizeof(SeedPair) * h_seed_pairs.size()));
    // Allocate a minimum of num_seed_pairs as all seed_pairs could be hsps in the worst case
    int32_t h_num_hsps = 0;
    ScoredSegmentPair* d_hsps;
    GW_CU_CHECK_ERR(cudaMalloc((void**)&d_hsps, sizeof(ScoredSegmentPair) * h_seed_pairs.size()));

    // Async Memcopy all the input values to device
    // TODO - Convert to pinned memory for true async copy
    GW_CU_CHECK_ERR(cudaMemcpyAsync(d_query, h_encoded_query, sizeof(char) * query_sequence.length(),
                                    cudaMemcpyHostToDevice, stream0.get()));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(d_target, h_encoded_target, sizeof(char) * target_sequence.length(),
                                    cudaMemcpyHostToDevice, stream0.get()));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(d_seed_pairs, &h_seed_pairs[0], sizeof(SeedPair) * h_seed_pairs.size(), cudaMemcpyHostToDevice,
                                    stream0.get()));

    // Create an ungapped extender object
    std::unique_ptr<Extender> ungapped_extender = create_extender(score_matrix, NUC2, xdrop_threshold, input_no_entropy, stream0.get(), 0);

    // Launch the ungapped extender device function
    ungapped_extender->extend_async(d_query, // Type TBD based on encoding
                                    query_sequence.length(),
                                    d_target,
                                    target_sequence.length(),
                                    score_threshold,
                                    d_seed_pairs,
                                    h_seed_pairs.size(),
                                    d_hsps,
                                    h_num_hsps);

    // Wait for ungapped extender to finish
    GW_CU_CHECK_ERR(cudaStreamSynchronize(stream0.get()));

    //Get results
    std::cout << "h_num_hsps=" << h_num_hsps << std::endl;

    free(h_encoded_target);
    free(h_encoded_query);
    // Free all allocated memory on the GPU
    GW_CU_CHECK_ERR(cudaFree(d_query));
    GW_CU_CHECK_ERR(cudaFree(d_target));
    GW_CU_CHECK_ERR(cudaFree(d_hsps));
    GW_CU_CHECK_ERR(cudaFree(d_seed_pairs));

    return 0;
}
