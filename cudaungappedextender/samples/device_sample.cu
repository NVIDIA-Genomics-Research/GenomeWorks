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

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/cudaungappedextender/cudaungappedextender.hpp>


using namespace claraparabricks::genomeworks;
using namespace claraparabricks::genomeworks::cudaungappedextender;

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
    
    std::string query_file_path = "../data/example.fa";
    std::unique_ptr<io::FastaParser> fasta_parser_query = io::create_kseq_fasta_parser(query_file_path, 0, false);
    // Assumes that only one sequence is present per file
    magic_sequence query_sequence = fasta_parser_query->get_sequence_by_id(0); 
    
    // CSV Anchors file - Each row -> query_position_in_read_, target_position_in_read_
    std::string anchors_file_path = "../data/example_hits.csv";
    
    std::vector<Anchor> h_hits;
    // Following function loops through all hits in the anchors.csv and returns results in 
    // the passed array
    parse_anchors(anchors_file_path, h_hits);

    // Following sections TBD based on encoding
    ScoreMatrix = magic_number_matrix;
    std::string h_encoded_target = magic_encode(magic_base, target_sequence);
    std::string h_encoded_query =  magic_encode(magic_base, query_sequence);

    // Create a stream for async use
    cudaStream_t stream0;
    cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
    // Allocate space on device for target and query sequences, hits,
    // high scoring segment pairs (hsps) and num_hsps.
    char* d_query;
    GW_CU_CHECK_ERROR(cudaMalloc((void**)&d_query, sizeof(char)*h_encoded_query.size())); 
    char* d_target;
    GW_CU_CHECK_ERROR(cudaMalloc((void**)&d_target, sizeof(char)*h_target_query.size())); 
    Anchor* d_hits;
    GW_CU_CHECK_ERROR(cudaMalloc((void**)&d_hits, sizeof(Anchor)*h_hits.size())); 
    // Allocate a minimum of num_hits as all hits could be hsps in the worst case
    int32_t h_num_hsps = 0;
    ScoredSegment* d_hsps;
    GW_CU_CHECK_ERROR(cudaMalloc((void**)&d_hsps, sizeof(ScoredSegment)*h_hits.size()));
    int32_t* d_num_hsps;
    GW_CU_CHECK_ERROR(cudaMalloc((void**)&d_num_hsps, sizeof(int32_t));

    // Async Memcopy all the input values to device
    GW_CU_CHECK_ERR(cudaMemcpyAsync(d_query, h_encoded_query.c_str(), sizeof(char)*h_encoded_query.size(), cudaMemcpyHostToDevice, stream0));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(d_target, h_encoded_target.c_str(), sizeof(char)*h_encoded_target.size(), cudaMemcpyHostToDevice, stream0));
    GW_CU_CHECK_ERR(cudaMemcpyAsync(d_hits, &h_hits[0], sizeof(Anchor)*h_hits.size(), cudaMemcpyHostToDevice, stream0));
    
    // Create an ungapped extender object
    std::unique_ptr<UngappedExtender> ungapped_extender = std::make_unique<UngappedExtender>(0, 
                                                                                                magic_number_matrix, 
                                                                                                input_xdrop,
                                                                                                input_no_entropy,
                                                                                                stream0);
    // Launch the ungapped extender device function
    ungapped_extender->ungapped_extend(d_query,  // Type TBD based on encoding
                                        encoded_query.size(),
                                        d_target.c_str(),
                                        encoded_target.size(),
                                        hsp_threshold,
                                        d_hits,
                                        h_hits.size(),
                                        d_hsps,
                                        d_num_hsps);
    // Copy back the number of hsps to host
    GW_CU_CHECK_ERR(cudaMemcpyAsync(&h_num_hsps, d_num_hsps, sizeof(int32_t), cudaMemcpyDeviceToHost, stream0));
    
    // Wait for ungapped extender to finish
    GW_CU_CHECK_ERR(cudaStreamSynchronize(stream0));
    
    //Get results
    if(h_num_hsps > 0)
    {
        std::vector<ScoredSegment> h_hsps(h_num_hsps);
        // Don't care about asynchronous copies here
        GW_CU_CHECK_ERR(cudaMemcpy(&h_hsps[0], d_hsps, sizeof(ScoredSegment)*h_num_hsps, cudaMemcpyDeviceToHost)); 
        
        int32_t i = 0;
        for (const auto& segment : h_hsps)
        {
            std::cout << "Segment: " << i << "Length: " << segment.len << "Score: " << segment.score << std::endl;
            std::cout << "Position in query: " << segment.anchor.query_position_in_read_<<std::endl;
            std::cout << "Position in target: " << segment.anchor.target_position_in_read_<<std::endl;
            i++;
        }
    
    }

    return 0;
    
}
