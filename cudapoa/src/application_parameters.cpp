/*
* Copyright 2019-2020 NVIDIA CORPORATION.
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

#include "application_parameters.hpp"
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>

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
        {"msa", no_argument, 0, 'a'},
        {"band-mode", required_argument, 0, 'b'},
        {"band-width", required_argument, 0, 'w'},
        {"dot", required_argument, 0, 'd'},
        {"max-groups", required_argument, 0, 'M'},
        {"gpu-mem-alloc", required_argument, 0, 'R'},
        {"match", required_argument, 0, 'm'},
        {"mismatch", required_argument, 0, 'n'},
        {"gap", required_argument, 0, 'g'},
        {"version", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
    };

    std::string optstring = "i:ab:w:d:M:R:m:n:g:vh";

    int32_t argument = 0;
    while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1)
    {
        switch (argument)
        {
        case 'i':
            input_paths.push_back(std::string(optarg));
            break;
        case 'a':
            msa = true;
            break;
        case 'b':
            if (std::stoi(optarg) < 0 || std::stoi(optarg) > 2)
            {
                throw std::runtime_error("band-mode must be either 0 for full bands, 1 for static bands or 2 for adaptive bands");
            }
            band_mode = static_cast<BandMode>(std::stoi(optarg));
            break;
        case 'w':
            band_width = std::stoi(optarg);
            break;
        case 'd':
            graph_output_path = std::string(optarg);
            break;
        case 'M':
            max_groups = std::stoi(optarg);
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
        throw std::runtime_error("gpu-mem-alloc must be greater than 0 and less than or equal to 1.0");
    }

    if (band_mode != BandMode::adaptive_band && band_width < 1)
    {
        throw std::runtime_error("band-width must be positive");
    }

    if (match_score < 0)
    {
        throw std::runtime_error("match score must be positive");
    }

    if (max_groups == 0)
    {
        throw std::runtime_error("max-groups cannot be 0");
    }

    if (mismatch_score > 0)
    {
        throw std::runtime_error("mismatch score must be non-positive");
    }

    if (gap_score > 0)
    {
        throw std::runtime_error("gap score must be non-positive");
    }

    verify_input_files(input_paths);
}

void ApplicationParameters::verify_input_files(std::vector<std::string>& inputpaths)
{
    // Checks if the files are either all fasta or if one file is provided, it needs to be fasta or cudapoa
    all_fasta = true;
    for (auto& file_path : inputpaths)
    {
        std::ifstream infile(file_path.c_str());
        if (infile.good())
        {
            std::string firstLine;
            std::getline(infile, firstLine);
            if (firstLine.at(0) != '>')
                all_fasta = false;
        }
        else
        {
            throw std::runtime_error(std::string("Invalid input file: ") + file_path);
        }
    }
    if (inputpaths.size() == 0 || (!all_fasta && inputpaths.size() > 1))
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
        -a, --msa
            generates msa if this flag is passed [default: consensus])"
              << R"(
        -b, --band-mode  <int>
            selects banding mode, 0: full-alignment, 1: static band, 2: adaptive band [2])"
              << R"(
        -w, --band-width <int>
            band-width for banded alignment (must be multiple of 128) [256])"
              << R"(
        -d, --dot <file>
            output path for printing graph in DOT format [disabled])"
              << R"(
        -M, --max-groups  <int>
            maximum number of POA groups to create from file (-1 for all, > 0 for limited) [-1]
            repeats groups if less groups are present than specified)"
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
