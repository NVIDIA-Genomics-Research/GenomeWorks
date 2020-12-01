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

#include <getopt.h>
#include <iostream>
#include <string>

#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/cudamapper/utils.hpp>
#include <claraparabricks/genomeworks/version.hpp>

namespace claraparabricks
{

namespace genomeworks
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
        {"rescue-overlap-ends", no_argument, 0, 'R'},
        {"drop-fused-overlaps", no_argument, 0, 'D'},
        {"query-indices-in-host-memory", required_argument, 0, 'Q'},
        {"query-indices-in-device-memory", required_argument, 0, 'q'},
        {"target-indices-in-host-memory", required_argument, 0, 'C'},
        {"target-indices-in-device-memory", required_argument, 0, 'q'},
        {"version", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
    };

    std::string optstring = "k:w:d:m:i:t:F:a:r:l:b:z:RDQ:q:C:c:BSvh";

    bool target_indices_in_host_memory_set   = false;
    bool target_indices_in_device_memory_set = false;
    bool custom_filtering_parameter          = false;
    int32_t argument                         = 0;
    format                                   = OutputFormat::PAF;
    while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1)
    {
        switch (argument)
        {
        case 'k':
            kmer_size = std::stoi(optarg);
            break;
        case 'w':
            windows_size = std::stoi(optarg);
            break;
        case 'd':
            num_devices = std::stoi(optarg);
            break;
        case 'm':
#ifndef GW_ENABLE_CACHING_ALLOCATOR
            std::cerr << "ERROR: Argument -m / --max-cached-memory cannot be used without caching allocator" << std::endl;
            exit(1);
#endif
            max_cached_memory = std::stoi(optarg);
            break;
        case 'i':
            index_size = std::stoi(optarg);
            break;
        case 't':
            target_index_size = std::stoi(optarg);
            break;
        case 'F':
            filtering_parameter        = std::stod(optarg);
            custom_filtering_parameter = true;
            break;
        case 'a':
            alignment_engines = std::stoi(optarg);
            throw_on_negative(alignment_engines, "Number of alignment engines should be non-negative");
            break;
        case 'r':
            min_residues = std::stoi(optarg);
            break;
        case 'l':
            min_overlap_len = std::stoi(optarg);
            break;
        case 'b':
            min_bases_per_residue = std::stoi(optarg);
            break;
        case 'z':
            min_overlap_fraction = std::stof(optarg);
            break;
        case 'R':
            perform_overlap_end_rescue = true;
            break;
        case 'D':
            drop_fused_overlaps = true;
            break;
        case 'Q':
            query_indices_in_host_memory = std::stoi(optarg);
            break;
        case 'q':
            query_indices_in_device_memory = std::stoi(optarg);
            break;
        case 'C':
            target_indices_in_host_memory     = std::stoi(optarg);
            target_indices_in_host_memory_set = true;
            break;
        case 'c':
            target_indices_in_device_memory     = std::stoi(optarg);
            target_indices_in_device_memory_set = true;
            break;
        case 'S':
#ifndef GW_BUILD_HTSLIB
            throw std::runtime_error("ERROR: Argument -S cannot be used without htslib");
#endif
            format = OutputFormat::SAM;
            break;
        case 'B':
#ifndef GW_BUILD_HTSLIB
            throw std::runtime_error("ERROR: Argument -B cannot be used without htslib");
#endif
            format = OutputFormat::BAM;
            break;
        case 'v':
            print_version();
            exit(1);
        case 'h':
            help(0);
            exit(1);
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
        std::cerr << "-C / --target-indices-in-host-memory not set, using -Q / --query-indices-in-host-memory value: " << query_indices_in_host_memory << std::endl;
        target_indices_in_host_memory = query_indices_in_host_memory;
    }

    if (!target_indices_in_device_memory_set)
    {
        std::cerr << "-c / --target-indices-in-device-memory not set, using -q / --query-indices-in-device-memory value: " << query_indices_in_device_memory << std::endl;
        target_indices_in_device_memory = query_indices_in_device_memory;
    }

    if (target_indices_in_host_memory < target_indices_in_device_memory)
    {
        std::cerr << "-C / --target-indices-in-host-memory  has to be larger or equal than -c / --target-indices-in-device-memory" << std::endl;
        exit(1);
    }

    if (query_indices_in_host_memory < query_indices_in_device_memory)
    {
        std::cerr << "-Q / --query-indices-in-host-memory  has to be larger or equal than -q / --query-indices-in-device-memory" << std::endl;
        exit(1);
    }

    query_filepath  = std::string(argv[optind++]);
    target_filepath = std::string(argv[optind++]);

    if (query_filepath == target_filepath)
    {
        all_to_all        = true;
        target_index_size = index_size;
        std::cerr << "NOTE - Since query and target files are same, activating all_to_all mode. Query index size used for both files." << std::endl;
    }

    create_input_parsers(query_parser, target_parser);

    set_filtering_parameter(query_parser, target_parser, custom_filtering_parameter);

    max_cached_memory_bytes = get_max_cached_memory_bytes();
}

void ApplicationParameters::set_filtering_parameter(std::shared_ptr<io::FastaParser>& query_parser,
                                                    std::shared_ptr<io::FastaParser>& target_parser,
                                                    const bool custom_filtering_parameter = false)
{

    number_of_basepairs_t total_sequence_length                 = 0;
    const number_of_basepairs_t minimum_for_automatic_filtering = 500000; // Require at least 0.5Mbp of sequence for filtering by default
    number_of_reads_t query_index                               = 0;
    number_of_reads_t target_index                              = 0;
    while (total_sequence_length < minimum_for_automatic_filtering && query_index < query_parser->get_num_seqences())
    {
        total_sequence_length += get_size<number_of_basepairs_t>(query_parser->get_sequence_by_id(query_index).seq);
        ++query_index;
    }

    while (total_sequence_length < minimum_for_automatic_filtering && target_index < target_parser->get_num_seqences())
    {
        total_sequence_length += get_size<number_of_basepairs_t>(target_parser->get_sequence_by_id(target_index).seq);
        ++target_index;
    }

    if (total_sequence_length < minimum_for_automatic_filtering && !custom_filtering_parameter)
    {
        filtering_parameter = 1.0;
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

int64_t ApplicationParameters::get_max_cached_memory_bytes()
{
#ifdef GW_ENABLE_CACHING_ALLOCATOR
    int64_t max_cached_bytes = 0;
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
            length of window to use for minimizers [10])"
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
            Remove representations with frequency (sketch_elements_with_that_representation/total_sketch_elements) >= filtering_parameter. Filtering is disabled if filtering_parameter == 1.0 (Min = 0.0, Max = 1.0) [1e-5])"
              << R"(
        -a, --alignment-engines
            Number of alignment engines to use (per device) for generating CIGAR strings for overlap alignments. Default value 0 = no alignment to be performed. Typically 2-4 engines per device gives best perf.)"
              << R"(
        -r, --min-residues
            Minimum number of matching residues in an overlap (recommended: 1 - 10) [3])"
              << R"(
        -l, --min-overlap-length
            Minimum length for an overlap [250].)"
              << R"(
        -b, --min-bases-per-residue
            Minimum number of bases in overlap per match [1000].)"
              << R"(
        -z, --min-overlap-fraction
            Minimum ratio of overlap length to alignment length [0.8].)"
              << R"(
        -R, --rescue-overlap-ends
            Run a kmer-based procedure that attempts to extend overlaps at the ends of the query/target.)"
              << R"(
        -D, --drop-fused-overlaps
            Remove overlaps which are joined into larger overlaps during fusion.)"
              << R"(
        -Q, --query-indices-in-host-memory
            number of query indices to keep in host memory [10])"
              << R"(
        -q, --query-indices-in-device-memory
            number of query indices to keep in device memory [5])"
              << R"(
        -C, --target-indices-in-host-memory
            number of target indices to keep in host memory [10])"
              << R"(
        -c, --target-indices-in-device-memory
            number of target indices to keep in device memory [5])"
              << R"(
        -S
            print overlaps in SAM format"
              << R"(
        -B
            print overlaps in BAM format"
              << R"(
        -v, --version
            Version information)"
              << std::endl;

    exit(exit_code);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
