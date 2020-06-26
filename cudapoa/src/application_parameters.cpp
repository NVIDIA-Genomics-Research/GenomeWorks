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
        {"result", required_argument, 0, 'r'},
        {"full-alignment", no_argument, 0, 'f'},
        {"band-width", required_argument, 0, 'b'},
        {"dot", required_argument, 0, 'd'},
        {"max-windows", required_argument, 0, 'w'},
        {"gpu-mem-alloc", required_argument, 0, 'R'},
        {"match", required_argument, 0, 'm'},
        {"mismatch", required_argument, 0, 'n'},
        {"gap", required_argument, 0, 'g'},
        {"version", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
    };

    std::string optstring = "i:r:fb:d:w:R:m:n:g:vh";

    int32_t argument = 0;
    while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1)
    {
        switch (argument)
        {
        case 'i':
            input_paths.push_back(std::string(optarg));
            break;
        case 'r':
            result = std::stoi(optarg);
            break;
        case 'f':
            banded = false;
            break;
        case 'b':
            band_width = std::stoi(optarg);
            break;
        case 'd':
            graph_output_path = std::string(optarg);
            break;
        case 'w':
            max_windows = std::stoi(optarg);
            break;
        case 'R':
            gpu_mem_allocation = std::stod(optarg);
            break;
        case 'm':
            match_score = std::stoi(optarg);
            break;
        case 'g':
            gap_score = std::stoi(optarg);
            break;
        case 'n':
            mismatch_score = std::stoi(optarg);
            break;
        case 'v':
            print_version();
        case 'h':
            help(0);
        default:
            exit(1);
        }
    }

    if (gpu_mem_allocation <= 0 || gpu_mem_allocation > 1.0)
    {
        std::cerr << "gpu-mem-alloc should be greater than 0 and less than 1.0" << std::endl;
        exit(1);
    }

    if (banded && band_width < 1)
    {
        std::cerr << "band-width must be positive" << std::endl;
        exit(1);
    }

    if (result < 0 || result > 1)
    {
        std::cerr << "result can only be 0 (consensus) or 1 (msa)" << std::endl;
        exit(1);
    }

    if (match_score < 0)
    {
        std::cerr << "match score must be positive" << std::endl;
        exit(1);
    }

    if (max_windows == 0)
    {
        std::cerr << "max-windows cannot be 0" << std::endl;
        exit(1);
    }

    if (mismatch_score > 0)
    {
        std::cerr << "mismatch score must be non-positive" << std::endl;
        exit(1);
    }

    if (gap_score > 0)
    {
        std::cerr << "gap score must be non-positive" << std::endl;
        exit(1);
    }

    verify_input_files(input_paths);
}

void ApplicationParameters::verify_input_files(std::vector<std::string>& input_paths)
{
    // Checks if the files are either all fasta or if one file is provided, it needs to be fasta or cudapoa
    all_fasta = true;
    for (auto& file_path : input_paths)
    {
        std::ifstream infile(file_path.c_str());
        if (infile.good())
        {
            std::string firstLine;
            std::getline(infile, firstLine);
            if (firstLine.at(0) != '>')
                all_fasta = false;
            //break; TODO- Break here or not? It also provides input filepath verification @atadkase
        }
        else
        {
            std::cerr << "Invalid input file: " << file_path << std::endl;
            exit(1);
        }
    }
    if (input_paths.size() == 0 || (!all_fasta && input_paths.size() > 1))
    {
        std::cerr << "Invalid input. cudapoa needs input in either one cudapoa format file or in one/multiple fasta files." << std::endl;
        help(1);
    }
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
        -i, --input <file>
            input in fasta/cudapoa format, can be used multiple times for multiple fasta files, but supports only one cudapoa file)"
              << R"(
        -r, --result <int>
            consensus(0)/msa(1) [0])"
              << R"(
        -f, --full-alignment
            uses full alignment if this flag is passed [banded alignment])"
              << R"(
        -b, --band-width <int>
            band-width for banded alignment (must be multiple of 128) [256])"
              << R"(
        -d, --dot <file>
            output path for printing graph in DOT format [disabled])"
              << R"(
        -w, --max-windows  <int>
            maximum number of windows to use from file (-1 for all and >0 for limited) [-1])"
              << R"(
        -R, --gpu-mem-alloc <double>
            fraction of available GPU memory to be used for cudapoa [0.9])"
              << R"(
        -m, --match  <int>
            score for matching bases (must be positive) [8])"
              << R"(
        -n, --mismatch  <int>
            score for mismatching bases (must be non-positive) [-6])"
              << R"(
        -g, --gap  <int>
            score for gaps (must be non-positive) [-8])"
              << R"(
        -v, --version
            version information)"
              << R"(
        -h, --help
            prints usage)"
              << std::endl;

    exit(exit_code);
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
