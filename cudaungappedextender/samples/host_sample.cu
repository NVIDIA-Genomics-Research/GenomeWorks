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
#include <string>
#include <vector>
#include <iostream>
#include <cuda_runtime_api.h>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/cudaungappedextender/cudaungappedextender.hpp>


namespace claraparabricks
{
namespace genomeworks
{
namespace cudaungappedextender
{
    int main(int argc, char* argv[])
    {
        const int32_t input_xdrop = 10;
        const int32_t input_no_entropy = 0;
        const int32_t hsp_threshold = 20000; 
        // Fasta query and target files
        std::string target_file_path = "../data/example.fa";
        std::unique_ptr<io::FastaParser> fasta_parser_target = io::create_kseq_fasta_parser(target_file_path, 0, false);
        // Assumes that only one sequence is present per file
        std::string target_sequence = fasta_parser_target->get_sequence_by_id(0); 
        
        magic_sequence query_file_path = "../data/example.fa";
        std::unique_ptr<io::FastaParser> fasta_parser_query = io::create_kseq_fasta_parser(query_file_path, 0, false);
        // Assumes that only one sequence is present per file
        magic_sequence query_sequence = fasta_parser_query->get_sequence_by_id(0); 
        
        // CSV Anchors file - Each row -> query_position_in_read_, target_position_in_read_
        std::string anchors_file_path = "../data/anchors.csv";
        
        std::vector<Anchor> h_hits;
        // Following function loops through all hits in the anchors.csv and returns results in 
        // the passed array
        parse_anchors(anchors_file_path, h_hits);

        // Following sections TBD based on encoding
        ScoreMatrix = magic_number_matrix;
        std::string encoded_target =  magic_encode(magic_base, target_sequence);
        std::string encoded_query =  magic_encode(magic_base, query_sequence);

        // Create a stream for async use
        cudaStream_t stream0;
        cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
        // Create an ungapped extender object
        std::unique_ptr<UngappedExtender> ungapped_extender = std::make_unique<UngappedExtender>(0, 
                                                                                                 magic_number_matrix, 
                                                                                                 input_xdrop,
                                                                                                 input_no_entropy,
                                                                                                 stream0);
        // Launch the ungapped extender host function
        ungapped_extender->ungapped_extend(encoded_query.c_str(),  // Type TBD based on encoding
                                           encoded_query.size(),
                                           encoded_target.c_str(),
                                           encoded_target.size(),
                                           hsp_threshold,
                                           h_hits);
        
        // Wait for ungapped extender to finish
        ungapped_extender->sync_extensions();

        //Get results
        const std::vector<std::shared_ptr<ScoredSegment>>& segments = ungapped_extender->get_scored_segments();
        int32_t i = 0;
        for (const auto& segment : segments)
        {
            std::cout << "Segment: " << i << "Length: " << segment->len << "Score: " << segment->score << std::endl;
            std::cout << "Position in query: " << segment->anchor.query_position_in_read_<<std::endl;
            std::cout << "Position in target: " << segment->anchor.target_position_in_read_<<std::endl;
            i++;
        }
        
        
    }
}
} // namespace cudaungappedextender
} // namespace genomeworks
} // namespace claraparabricks
