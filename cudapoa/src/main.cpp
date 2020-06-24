/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#include <iostream>

#include <string>
#include <claraparabricks/genomeworks/cudapoa/utils.hpp>
#include <claragenomics/io/fasta_parser.hpp>


#include "application_parameters.hpp"


namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{



void generate_window_data(bool all_fasta, std::vector<std::string>& input_paths, std::vector<std::vector<std::string>>& windows)
{
    if(all_fasta)
    {
        const int32_t min_sequence_length    = 0;
        const int32_t num_input_files        = input_paths.size();
        std::vector<std::shared_ptr<claragenomics::io::FastaParser>> fasta_parser_vec(num_input_files);
        std::vector<int64_t> num_reads_per_file(num_input_files);
        for (int32_t i = 0; i < num_input_files; i++)
        {
            fasta_parser_vec[i]   = claragenomics::io::create_kseq_fasta_parser(input_paths[i], min_sequence_length, false);
            num_reads_per_file[i] = fasta_parser_vec[i]->get_num_seqences();
        }
        const int64_t num_reads = num_reads_per_file[0];
        
        for(int32_t i = 1; i < num_input_files; i++)
        {
            if(num_reads_per_file[i] != num_reads)
            {
                std::cerr << "Failed to read input files." << std::endl;
                std::cerr << "Number of long-reads per input file do not match with each other." << std::endl;
                assert(false);
                return;
            }
        }
        windows.resize(num_reads);
    
        int64_t idx = 0;
        for (auto& window: windows)
        {
            for(int32_t i = 0; i < num_input_files; i++)
            {
                window.push_back(fasta_parser_vec[i]->get_sequence_by_id(idx).seq);
            }
            idx++;
        }
    }
    else
    {
        parse_window_data_file(windows, input_paths[0], -1);
    }
    
}

int main(int argc, char* argv[])
{
    
    const ApplicationParameters parameters(argc, argv);


    return 0;
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks

/// \brief main function
/// main function cannot be in a namespace so using this function to call actual main function
int main(int argc, char* argv[])
{
    return claraparabricks::genomeworks::cudapoa::main(argc, argv);
}
