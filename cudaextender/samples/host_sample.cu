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
#include <claraparabricks/genomeworks/cudaungappedextender/cudaungappedextender.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <vector>

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
    // Following function loops through all seed_pairs in the SeedPairs csv and returns
    // results in
    // the passed vector
    parse_SeedPairs(seed_pairs_file_path, h_seed_pairs);

    // Following sections TBD based on encoding
    ScoreMatrix                = magic_number_matrix;
    std::string encoded_target = magic_encode(magic_base, target_sequence);
    std::string encoded_query  = magic_encode(magic_base, query_sequence);

    // Create a stream for async use
    CudaStream stream0 = make_cuda_stream();
    // Create an ungapped extender object
    std::unique_ptr<UngappedExtender> ungapped_extender =
        std::make_unique<UngappedExtender>(0, magic_number_matrix, input_xdrop,
                                           input_no_entropy, stream0.get());
    // Launch the ungapped extender host function
    ungapped_extender->extend_async(
        encoded_query.c_str(), // Type TBD based on encoding
        encoded_query.size(), encoded_target.c_str(), encoded_target.size(),
        score_threshold, h_seed_pairs);

    // Wait for ungapped extender to finish
    ungapped_extender->sync();

    // Get results
    const std::vector<ScoredSegmentPair>& segments =
        ungapped_extender->get_scored_segment_pairs();
    int32_t i = 0;
    for (const auto& segment : segments)
    {
        std::cout << "Segment: " << i << "Length: " << segment.length
                  << "Score: " << segment.score << std::endl;
        std::cout << "Position in query: "
                  << segment.seed_pair.query_position_in_read << std::endl;
        std::cout << "Position in target: "
                  << segment.seed_pair.target_position_in_read << std::endl;
        i++;
    }

    return 0;
}
