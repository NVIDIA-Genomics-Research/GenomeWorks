/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <chrono>
#include <getopt.h>
#include <iostream>
#include <string>

#include <claragenomics/logging/logging.hpp>

#include "cudamapper/index.hpp"
#include "matcher.hpp"
#include "overlapper_triggered.hpp"


static struct option options[] = {
        {"window-size", required_argument , 0, 'w'},
        {"kmer-size", required_argument, 0, 'k'},
        {"help", no_argument, 0, 'h'},
};

void help();

int main(int argc, char *argv[])
{
    claragenomics::logging::Init();

    uint32_t k = 15;
    uint32_t w = 15;
    std::string optstring = "k:w:h";
    uint32_t argument;
    while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1){
        switch (argument) {
            case 'k':
                k = atoi(optarg);
                break;
            case 'w':
                w = atoi(optarg);
                break;
            case 'h':
                help();
                exit(0);
            default:
                exit(1);
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    CGA_LOG_INFO("Creating index");

    if (optind >= argc){
        help();
        exit(1);
    }

    if (k > claragenomics::Index::maximum_kmer_size()){
        std::cerr << "kmer of size " << k << " is not allowed, maximum k = " <<
            claragenomics::Index::maximum_kmer_size() << std::endl;
        exit(1);
    }

    std::string input_filepath = std::string(argv[optind]);
    std::unique_ptr<claragenomics::Index> index = claragenomics::Index::create_index(input_filepath, k, w, 7, 7);
    CGA_LOG_INFO("Created index");
    std::cerr << "Index execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    auto num_reads = index.get()->number_of_reads();
    auto match_point = num_reads / 2; // Let's say half the reads are query and half are target.

    start_time = std::chrono::high_resolution_clock::now();
    CGA_LOG_INFO("Started matcher");
    claragenomics::Matcher matcher(*index, match_point);
    CGA_LOG_INFO("Finished matcher");
    std::cerr << "Matcher execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    start_time = std::chrono::high_resolution_clock::now();
    CGA_LOG_INFO("Started overlap detector");
    auto overlapper = claragenomics::OverlapperTriggered();
    auto overlaps = overlapper.get_overlaps(matcher.anchors(), *index);

    CGA_LOG_INFO("Finished overlap detector");
    std::cerr << "Overlap detection execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

    overlapper.print_paf(overlaps);
    return  0;
}

void help() {
    std::cout<<
    R"(Usage: cudamapper [options ...] <sequences>
     <sequences>
        Input file in FASTA/FASTQ format (can be compressed with gzip)
        containing sequences used for all-to-all overlapping
     options:
        -k, --kmer-size
            length of kmer to use for minimizers [15] (Max=)" << claragenomics::Index::maximum_kmer_size() << ")" << R"(
        -w, --window-size
            length of window to use for minimizers [15])" << std::endl;
}
