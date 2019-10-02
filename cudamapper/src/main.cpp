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

    //Now carry out all the looped polling
    size_t index_size = 40000;
    size_t query_start = 0;
    size_t query_end = query_start + index_size;

    std::string input_filepath = std::string(argv[optind]);

    while(true) { // outer loop over query

        //first generate a2a for query
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        std::pair<std::uint64_t, std::uint64_t> query_range {query_start, query_end};

        ranges.push_back(query_range);

        std::unique_ptr<claragenomics::Index> index = claragenomics::Index::create_index(input_filepath, k, w, ranges);

        CGA_LOG_INFO("Created index");
        std::cerr << "Index execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

        //auto num_reads = index.get()->number_of_reads();
        auto match_point = 0; // all to all

        start_time = std::chrono::high_resolution_clock::now();
        CGA_LOG_INFO("Started matcher");
        claragenomics::Matcher matcher(*index, match_point);
        CGA_LOG_INFO("Finished matcher");
        std::cerr << "Matcher execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        CGA_LOG_INFO("Started overlap detector");
        auto overlapper = claragenomics::OverlapperTriggered();
        auto overlaps = overlapper.get_overlaps(matcher.anchors(), *index);

        CGA_LOG_INFO("Finished overlap detector");
        std::cerr << "Overlap detection execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

        overlapper.print_paf(overlaps);

        //Now that the all-to-all overlaps for the query have been generated,
        //the first read in the targets s set to be the read after the last read in the query.
        size_t target_start = query_end + 1;
        size_t target_end = target_start + index_size;

        // No more reads to process.
        if (index.get()->reached_end_of_input()){
            break;
        }

        while(true){ //Now loop over the targets
            //first generate a2a for query
            std::vector<std::pair<std::uint64_t, std::uint64_t>> target_ranges;
            std::pair<std::uint64_t, std::uint64_t> target_range {target_start, target_end};

            target_ranges.push_back(query_range);
            target_ranges.push_back(target_range);

            auto new_index = claragenomics::Index::create_index(input_filepath, k, w, target_ranges);

            CGA_LOG_INFO("Created index");
            std::cerr << "Index execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

            // Match point is the index up to which all reads in the query are part of the index
            // We therefore set it to be the number of reads in the query (query read index end - query read index start)
            //The number of reads in the whole target chunk is set to be index size.
            match_point = (query_range.second - query_range.first);

            start_time = std::chrono::high_resolution_clock::now();
            CGA_LOG_INFO("Started matcher");
            claragenomics::Matcher qt_matcher(*new_index, match_point);
            CGA_LOG_INFO("Finished matcher");
            std::cerr << "Matcher execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

            start_time = std::chrono::high_resolution_clock::now();
            CGA_LOG_INFO("Started overlap detector");
            //auto overlapper = claragenomics::OverlapperTriggered();
            overlaps = overlapper.get_overlaps(qt_matcher.anchors(), *new_index);

            CGA_LOG_INFO("Finished overlap detector");
            std::cerr << "Overlap detection execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

            overlapper.print_paf(overlaps);

            if (new_index.get()->number_of_reads()  < (index_size * 2)){ //reached the end of the reads
                break;
            }

            //Now that mappings from query to one range of targets has been completed,
            //the new target start is set to be the next read index after the last read
            //from the previous chunk
            //The number of reads in the whole target chunk is set to be index size.
            target_start = target_end + 1;
            target_end = target_start + index_size;
        }
        //update query positions
        query_start = query_end + 1;
        query_end = query_start + index_size;
    }
    return 0;
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
