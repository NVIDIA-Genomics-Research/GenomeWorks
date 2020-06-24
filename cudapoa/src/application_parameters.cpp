/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "application_parameters.hpp"
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/version.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

ApplicationParameters::ApplicationParameters(int argc, char* argv[])
{
    struct option options[] = {
        {"input", required_argument, 0, 'i'},
        {"mode", required_argument, 0, 'm'},
        {"full-alignment", no_argument, 0, 'f'},
        {"band-width", required_argument, 0, 'b'},
        {"graph-output", required_argument, 0, 'g'},
        {"batch-size", required_argument, 0, 's'},
        {"max-reads-per-window", required_argument, -0, 'n'},
        {"gpu-mem-alloc", required_argument, 0, 'r'},
        {"version", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
    };

    std::string optstring = "m:fb:g:s:n:r:vh";

    int32_t argument                         = 0;
    while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1)
    {
        switch (argument)
        {
        case 'i':
            input_paths.push_back(std::string(optarg));
        case 'm':
            consensus_mode = std::stoi(optarg);
            break;
        case 'f':
            full_alignment = true;
            break;
        case 'b':
            band_width = std::stoi(optarg);
            break;
        case 'g':
            graph_output_path = std::string(optarg);
            break;
        case 's':
            batch_size = std::stoi(optarg);
            break;
        case 'n':
            max_reads_per_window = std::stoi(optarg);
            break;
        case 'd':
            gpu_mem_allocation = std::stod(optarg);
            break;
        case 'v':
            print_version();
        case 'h':
            help(0);
        default:
            exit(1);
        }
    }

    verify_input_files(input_paths);

}

void ApplicationParameters::verify_input_files(std::vector<std::string>& input_paths)
{
    // Checks if the files are either all fasta or if one file is provided, it needs to be fasta or cudapoa
    all_fasta = true;
    for(auto & file_path: input_paths)
    {
        std::ifstream infile(file_path.c_str());
        if(infile.good())
        {
            std::string firstLine;
            std::getline(infile, firstLine);
            if (firstLine.at(0) != '>')
                all_fasta = false;
                //break; TODO- Break here or not? It also provides input filepath verification @atadkase
        }
        else
        {
            std::cerr << "Invalid input file: "<< file_path << std::endl;
            exit(1);
        }
        
    }
    if(input_paths.size() == 0 || (!all_fasta && input_paths.size() > 1))
    {
        std::cerr<<"Invalid input. cudapoa needs input in either one cudapoa format file or in one/multiple fasta files."<<std::endl;
        exit(1);
    }
}

void ApplicationParameters::create_input_parsers(std::shared_ptr<io::FastaParser>& query_parser,
                                                 std::shared_ptr<io::FastaParser>& target_parser)
{
    assert(query_parser == nullptr);
    assert(target_parser == nullptr);

    query_parser = io::create_kseq_fasta_parser(query_filepath, kmer_size + windows_size - 1);

    if (all_to_all)
    {
        target_parser = query_parser;
    }
    else
    {
        target_parser = io::create_kseq_fasta_parser(target_filepath, kmer_size + windows_size - 1);
    }

    std::cerr << "Query file: " << query_filepath << ", number of reads: " << query_parser->get_num_seqences() << std::endl;
    std::cerr << "Target file: " << target_filepath << ", number of reads: " << target_parser->get_num_seqences() << std::endl;
}

void ApplicationParameters::print_version(const bool exit_on_completion)
{
    std::cerr << claraparabricks_genomeworks_version() << std::endl;

    if (exit_on_completion)
    {
        exit(1);
    }
}

void ApplicationParameters::help(int32_t exit_code)
{
    std::cerr <<
        R"(Usage: cudapoa [options ...]
     options:
        -i, --input
            input in fasta/cudapoa format, can be used multiple times for multiple fasta files, but supports only one cudapoa file)"
              << R"(
        -m, --mode
            consensus(0)/msa(1) [0])"
              << R"(
        -f, --full-alignment
            uses full alignment if this flag is passed [banded alignment])"
              << R"(
        -g, --graph-output
            output path for printing graph in DOT format [disabled])"
              << R"(
        -b, --batch-size
            length of batch size used for cudapoa batch [30])" // TODO - @atadkase
              << R"(
        -n, --max-reads-per-window
            maximum number of reads to use per window)"
              << R"(
        -r, --gpu-mem-alloc
            fraction of GPU memory to be used for cudapoa [0.9])"
              << R"(
        -v, --version
            Version information)"
              << std::endl;

    exit(exit_code);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
