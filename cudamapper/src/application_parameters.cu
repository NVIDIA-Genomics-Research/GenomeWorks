/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "application_parameters.cuh"

#include <getopt.h>
#include <iostream>

#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/io/fasta_parser.hpp>

namespace claragenomics
{
namespace cudamapper
{

ApplicationParameters::ApplicationParameters(int argc, char* argv[])
{
    struct option options[] = {
        {"kmer-size", required_argument, 0, 'k'},
        {"window-size", required_argument, 0, 'w'},
        {"num-devices", required_argument, 0, 'd'},
        {"max-cached-memory", required_argument, 0, 'm'},
        {"index-size", required_argument, 0, 'i'},
        {"target-index-size", required_argument, 0, 't'},
        {"filtering-parameter", required_argument, 0, 'F'},
        {"alignment-engines", required_argument, 0, 'a'},
        {"min-residues", required_argument, 0, 'r'},
        {"min-overlap-length", required_argument, 0, 'l'},
        {"min-bases-per-residue", required_argument, 0, 'b'},
        {"min-overlap-fraction", required_argument, 0, 'z'},
        {"query-indices-in-host-memory", required_argument, 0, 'q'},
        {"query-indices-in-device-memory", required_argument, 0, 'Q'},
        {"target-indices-in-host-memory", required_argument, 0, 'c'},
        {"target-indices-in-device-memory", required_argument, 0, 'C'},
        {"help", no_argument, 0, 'h'},
    };

    std::string optstring = "k:w:d:m:i:t:F:h:a:r:l:b:z:q:Q:c:C:";

    bool target_indices_in_host_memory_set   = false;
    bool target_indices_in_device_memory_set = false;
    int32_t argument                         = 0;
    while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1)
    {
        switch (argument)
        {
        case 'k':
            kmer_size = atoi(optarg);
            break;
        case 'w':
            windows_size = atoi(optarg);
            break;
        case 'd':
            num_devices = atoi(optarg);
            break;
        case 'm':
#ifndef CGA_ENABLE_CACHING_ALLOCATOR
            std::cerr << "ERROR: Argument -m / --max-cached-memory cannot be used without caching allocator" << std::endl;
            exit(1);
#endif
            max_cached_memory = atoi(optarg);
            break;
        case 'i':
            index_size = atoi(optarg);
            break;
        case 't':
            target_index_size = atoi(optarg);
            break;
        case 'F':
            filtering_parameter = atof(optarg);
            break;
        case 'a':
            alignment_engines = atoi(optarg);
            throw_on_negative(alignment_engines, "Number of alignment engines should be non-negative");
            break;
        case 'r':
            min_residues = atoi(optarg);
            break;
        case 'l':
            min_overlap_len = atoi(optarg);
            break;
        case 'b':
            min_bases_per_residue = atoi(optarg);
            break;
        case 'z':
            min_overlap_fraction = atof(optarg);
            break;
        case 'q':
            query_indices_in_host_memory = atoi(optarg);
            break;
        case 'Q':
            query_indices_in_device_memory = atoi(optarg);
            break;
        case 'c':
            target_indices_in_host_memory     = atoi(optarg);
            target_indices_in_host_memory_set = true;
            break;
        case 'C':
            target_indices_in_device_memory     = atoi(optarg);
            target_indices_in_device_memory_set = true;
            break;
        case 'h':
            help(0);
        default:
            exit(1);
        }
    }

    if (kmer_size > Index::maximum_kmer_size())
    {
        std::cerr << "kmer of size " << kmer_size << " is not allowed, maximum k = " << Index::maximum_kmer_size() << std::endl;
        exit(1);
    }

    if (filtering_parameter > 1.0 || filtering_parameter < 0.0)
    {
        std::cerr << "-F / --filtering-parameter must be in range [0.0, 1.0]" << std::endl;
        exit(1);
    }

    if (max_cached_memory < 0)
    {
        std::cerr << "-m / --max-cached-memory must not be negative" << std::endl;
        exit(1);
    }

    // Check remaining argument count.
    if ((argc - optind) < 2)
    {
        std::cerr << "Invalid inputs. Please refer to the help function." << std::endl;
        help(1);
    }

    if (!target_indices_in_host_memory_set)
    {
        std::cerr << "-c / --target-indices-in-host-memory not set, using -q / --query-indices-in-host-memory value: " << query_indices_in_host_memory << std::endl;
        target_indices_in_host_memory = query_indices_in_host_memory;
    }

    if (!target_indices_in_device_memory_set)
    {
        std::cerr << "-C / --target-indices-in-device-memory not set, using -Q / --query-indices-in-device-memory value: " << query_indices_in_device_memory << std::endl;
        target_indices_in_device_memory = query_indices_in_device_memory;
    }

    query_filepath  = std::string(argv[optind++]);
    target_filepath = std::string(argv[optind++]);

    if (query_filepath == target_filepath)
    {
        all_to_all        = true;
        target_index_size = index_size;
        std::cerr << "NOTE - Since query and target files are same, activating all_to_all mode. Query index size used for both files." << std::endl;
    }

    get_input_parsers(query_parser, target_parser);

    max_cached_memory_bytes = get_max_cached_memory_bytes();
}

void ApplicationParameters::get_input_parsers(std::shared_ptr<io::FastaParser>& query_parser,
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

std::size_t ApplicationParameters::get_max_cached_memory_bytes()
{
#ifdef CGA_ENABLE_CACHING_ALLOCATOR
    std::size_t max_cached_bytes = 0;
    if (max_cached_memory == 0)
    {
        std::cerr << "Programmatically looking for max cached memory" << std::endl;
        max_cached_bytes = cudautils::find_largest_contiguous_device_memory_section();

        if (max_cached_bytes == 0)
        {
            std::cerr << "No memory available for caching" << std::endl;
            exit(1);
        }
    }
    else
    {
        max_cached_bytes = max_cached_memory * 1024ull * 1024ull * 1024ull; // max_cached_memory is in GiB
    }

    std::cerr << "Using device memory cache of " << max_cached_bytes << " bytes" << std::endl;

    return max_cached_bytes;
#else
    return 0;
#endif
}

void ApplicationParameters::help(int32_t exit_code)
{
    std::cerr <<
        R"(Usage: cudamapper [options ...] <query_sequences> <target_sequences>
     <sequences>
        Input file in FASTA/FASTQ format (can be compressed with gzip)
        containing sequences used for all-to-all overlapping
     options:
        -k, --kmer-size
            length of kmer to use for minimizers [15] (Max=)"
              << Index::maximum_kmer_size() << ")"
              << R"(
        -w, --window-size
            length of window to use for minimizers [15])"
              << R"(
        -d, --num-devices
            number of GPUs to use [1])"
              << R"(
        -m, --max-cached-memory
            maximum aggregate cached memory per device in GiB, if 0 program tries to allocate as much memory as possible [0])"
              << R"(
        -i, --index-size
            length of batch size used for query in MB [30])"
              << R"(
        -t, --target-index-size
            length of batch sized used for target in MB [30])"
              << R"(
        -F, --filtering-parameter
            filter all representations for which sketch_elements_with_that_representation/total_sketch_elements >= filtering_parameter), filtering disabled if filtering_parameter == 1.0 [1'000'000'001] (Min = 0.0, Max = 1.0))"
              << R"(
        -a, --alignment-engines
            Number of alignment engines to use (per device) for generating CIGAR strings for overlap alignments. Default value 0 = no alignment to be performed. Typically 2-4 engines per device gives best perf.)"
              << R"(
        -r, --min-residues
            Minimum number of matching residues in an overlap [10])"
              << R"(
        -l, --min-overlap-length
            Minimum length for an overlap [500].)"
              << R"(
        -b, --min-bases-per-residue
            Minimum number of bases in overlap per match [100].)"
              << R"(
        -z, --min-overlap-fraction
            Minimum ratio of overlap length to alignment length [0.95].)
              << R"(
        -q, --query-indices-in-host-memory
            number of query indices to keep in host memory [10])"
              << R"(
        -Q, --query-indices-in-device-memory
            number of query indices to keep in device memory [5])"
              << R"(
        -c, --target-indices-in-host-memory
            number of target indices to keep in host memory [10])"
              << R"(
        -C, --target-indices-in-device-memory
            number of target indices to keep in device memory [5])"
              << std::endl;

    exit(exit_code);
}

} // namespace cudamapper
} // namespace claragenomics
