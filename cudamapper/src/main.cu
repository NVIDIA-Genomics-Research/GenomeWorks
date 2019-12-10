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
#include <deque>
#include <mutex>
#include <future>
#include <thread>
#include <atomic>
#include <map>

#include "ThreadPool.h"

#include <claragenomics/logging/logging.hpp>
#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/cudautils.hpp>

#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/cudamapper/matcher.hpp>
#include <claragenomics/cudamapper/overlapper.hpp>
#include "overlapper_triggered.hpp"

static struct option options[] = {
    {"window-size", required_argument, 0, 'w'},
    {"kmer-size", required_argument, 0, 'k'},
    {"num-devices", required_argument, 0, 'd'},
    {"index-size", required_argument, 0, 'i'},
    {"target-index-size", required_argument, 0, 't'},
    {"max-cache-size", required_argument, 0, 'c'},
    {"help", no_argument, 0, 'h'},
};

void help(int32_t exit_code);

int main(int argc, char* argv[])
{
    claragenomics::logging::Init();

    uint32_t k                     = 15;
    uint32_t w                     = 15;
    std::int32_t index_size        = 10000;
    std::int32_t num_devices       = 1;
    std::int32_t target_index_size = 10000;
    std::int32_t max_cache_size    = 100;
    std::string optstring          = "t:i:k:w:h:d:c:";
    uint32_t argument;
    while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1)
    {
        switch (argument)
        {
        case 'k':
            k = atoi(optarg);
            break;
        case 'w':
            w = atoi(optarg);
            break;
        case 'i':
            index_size = atoi(optarg);
            break;
        case 'd':
            num_devices = atoi(optarg);
            break;
        case 't':
            target_index_size = atoi(optarg);
            break;
        case 'c':
            max_cache_size = atoi(optarg);
            break;
        case 'h':
            help(0);
        default:
            exit(1);
        }
    }

    if (k > claragenomics::cudamapper::Index::maximum_kmer_size())
    {
        std::cerr << "kmer of size " << k << " is not allowed, maximum k = " << claragenomics::cudamapper::Index::maximum_kmer_size() << std::endl;
        exit(1);
    }

    // Check remaining argument count.
    if ((argc - optind) < 2)
    {
        std::cerr << "Invalid inputs. Please refer to the help function." << std::endl;
        help(1);
    }

    std::string query_filepath  = std::string(argv[optind++]);
    std::string target_filepath = std::string(argv[optind++]);

    bool all_to_all = false;
    if (query_filepath == target_filepath)
    {
        all_to_all        = true;
        target_index_size = index_size;
        std::cerr << "NOTE - Since query and target files are same, activating all_to_all mode. Query index size used for both files." << std::endl;
    }

    std::unique_ptr<claragenomics::io::FastaParser> query_parser = claragenomics::io::create_fasta_parser(query_filepath);
    int32_t queries                                              = query_parser->get_num_seqences();

    std::unique_ptr<claragenomics::io::FastaParser> target_parser = claragenomics::io::create_fasta_parser(target_filepath);
    int32_t targets                                               = target_parser->get_num_seqences();

    std::cerr << "Query " << query_filepath << " index " << queries << std::endl;
    std::cerr << "Target " << target_filepath << " index " << targets << std::endl;

    // Data structure for holding overlaps to be written out
    std::mutex overlaps_writer_mtx;
    std::mutex index_cache_mtx;

    struct query_target_range
    {
        std::pair<std::int32_t, int32_t> query_range;
        std::vector<std::pair<std::int32_t, int32_t>> target_ranges;
    };

    //First generate all the ranges independently, then loop over them.
    std::vector<query_target_range> query_target_ranges;

    for (std::int32_t query_start_index = 0; query_start_index < queries; query_start_index += index_size)
    {

        std::int32_t query_end_index = std::min(query_start_index + index_size, queries);

        query_target_range q;
        q.query_range = std::make_pair(query_start_index, query_end_index);

        std::int32_t target_start_index = 0;
        // If all_to_all mode, then we can optimzie by starting the target sequences from the same index as
        // query because all indices before the current query index are guaranteed to have been processed in
        // a2a mapping.
        if (all_to_all)
        {
            target_start_index = query_start_index;
        }

        for (; target_start_index < targets; target_start_index += target_index_size)
        {
            std::int32_t target_end_index = std::min(target_start_index + target_index_size,
                                                     targets);
            q.target_ranges.push_back(std::make_pair(target_start_index, target_end_index));
        }

        query_target_ranges.push_back(q);
    }

    // This is a per-device cache, if it has the index it will return it, if not it will generate it, store and return it.
    std::vector<std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<claragenomics::cudamapper::Index>>> index_cache(num_devices);

    auto get_index = [&index_cache, &index_cache_mtx, max_cache_size](claragenomics::io::FastaParser& parser,
                                                                      const claragenomics::cudamapper::read_id_t start_index,
                                                                      const claragenomics::cudamapper::read_id_t end_index,
                                                                      const std::uint64_t k,
                                                                      const std::uint64_t w,
                                                                      const int device_id,
                                                                      const bool all_to_all) {
        CGA_NVTX_RANGE(profiler, "get index");
        std::pair<uint64_t, uint64_t> key;
        key.first  = start_index;
        key.second = end_index;

        std::shared_ptr<claragenomics::cudamapper::Index> index;

        if (index_cache[device_id].count(key))
        {
            index = index_cache[device_id][key];
        }
        else
        {
            index = std::move(claragenomics::cudamapper::Index::create_index(parser, start_index, end_index, k, w));

            // If in all-to-all mode, put this query in the cache for later use.
            // Cache eviction is handled later on by the calling thread
            // using the evict_index function.
            if (index_cache[device_id].size() < max_cache_size && all_to_all)
            {
                index_cache_mtx.lock();
                index_cache[device_id][key] = index;
                index_cache_mtx.unlock();
            }
        }
        return index;
    };

    // When performing all-to-all mapping, indices are instantitated as start-end-ranges in the reads.
    // As such, once a query index has been used it will not be needed again. For example, parsing ranges
    // [0-999], [1000-1999], [2000-2999], the caching/eviction would be as follows:
    //
    // Round 1
    // Query: [0-999] - Enter cache
    // Target: [1000-1999] - Enter cache
    // Target: [1999 - 2999] - Enter cache
    // Evict [0-999]
    // Round 2
    // Query: [1000-1999] - Use cache entry (from previous use when now query was a target)
    // Etc..
    auto evict_index = [&index_cache, &index_cache_mtx](const claragenomics::cudamapper::read_id_t query_start_index,
                                                        const claragenomics::cudamapper::read_id_t query_end_index,
                                                        const int device_id) {
        std::pair<uint64_t, uint64_t> key;
        key.first  = query_start_index;
        key.second = query_end_index;

        index_cache_mtx.lock();
        index_cache[device_id].erase(key);
        index_cache_mtx.unlock();
    };

    auto compute_overlaps = [&](const query_target_range query_target_range, const int device_id) {
        std::vector<std::shared_ptr<std::future<void>>> print_pafs_futures;

        cudaSetDevice(device_id);

        auto query_start_index = query_target_range.query_range.first;
        auto query_end_index   = query_target_range.query_range.second;

        std::cerr << "Procecssing query range: (" << query_start_index << " - " << query_end_index - 1 << ")" << std::endl;

        std::shared_ptr<claragenomics::cudamapper::Index> query_index(nullptr);
        std::shared_ptr<claragenomics::cudamapper::Index> target_index(nullptr);
        std::unique_ptr<claragenomics::cudamapper::Matcher> matcher(nullptr);

        {
            CGA_NVTX_RANGE(profiler, "generate_query_index");
            auto start_time = std::chrono::high_resolution_clock::now();

            query_index = get_index(*query_parser, query_start_index, query_end_index, k, w, device_id, all_to_all);
        }

        //Main loop
        for (const auto target_range : query_target_range.target_ranges)
        {

            auto target_start_index = target_range.first;
            auto target_end_index   = target_range.second;

            {
                CGA_NVTX_RANGE(profiler, "generate_target_index");
                auto start_time = std::chrono::high_resolution_clock::now();
                target_index    = get_index(*target_parser, target_start_index, target_end_index, k, w, device_id, all_to_all);
            }
            {
                CGA_NVTX_RANGE(profiler, "generate_matcher");
                auto start_time = std::chrono::high_resolution_clock::now();
                matcher         = claragenomics::cudamapper::Matcher::create_matcher(*query_index,
                                                                             *target_index);
            }
            {

                claragenomics::cudamapper::OverlapperTriggered overlapper;
                CGA_NVTX_RANGE(profiler, "generate_overlaps");
                auto start_time = std::chrono::high_resolution_clock::now();

                // Get unfiltered overlaps
                std::vector<claragenomics::cudamapper::Overlap> overlaps_to_add;
                overlapper.get_overlaps(overlaps_to_add, matcher->anchors(), *query_index, *target_index);

                std::shared_ptr<std::future<void>> write_and_filter_overlaps_future = std::make_shared<std::future<void>>(std::async(std::launch::async,
                                                                                                                                     [&overlaps_writer_mtx, overlaps_to_add](std::vector<claragenomics::cudamapper::Overlap> overlaps) {
                                                                                                                                         std::vector<claragenomics::cudamapper::Overlap> filtered_overlaps;
                                                                                                                                         claragenomics::cudamapper::Overlapper::filter_overlaps(filtered_overlaps, overlaps_to_add);
                                                                                                                                         overlaps_writer_mtx.lock();
                                                                                                                                         claragenomics::cudamapper::Overlapper::print_paf(filtered_overlaps);
                                                                                                                                         overlaps_writer_mtx.unlock();
                                                                                                                                     },
                                                                                                                                     overlaps_to_add));

                print_pafs_futures.push_back(write_and_filter_overlaps_future);
            }
        }

        //Query will no longer be needed on device, remove it from the cache
        evict_index(query_start_index, query_end_index, device_id);
        return print_pafs_futures;
    };

    // The application (File parsing, index generation, overlap generation etc) is all launched from here.
    // The main application works as follows:
    // 1. Generate a thread pool with one worker thread per device (GPU).
    // 2. For each worker, assign a query-target-range, this is list of query reads and a *list of lists of target reads* to match against.
    // 3. Each worker returns a vector of futures (since overlap writing is dispatched to an async thread on host). All futures are waited for before the main application exits.

    // 1. create thread pool to compute overlaps. One worker thread per device.
    ThreadPool overlap_pool(num_devices);

    // 2. Enqueue the query-target ranges which need to be computed, each thread returns a vector of futures for the threads it launches
    std::vector<std::future<std::vector<std::shared_ptr<std::future<void>>>>> overlap_futures;
    for (int i = 0; i < query_target_ranges.size(); i++)
    {
        // assign chunk pairs (Query, target) to device
        auto query_target_range = query_target_ranges[i];
        auto device_id          = i % num_devices;
        overlap_futures.push_back(overlap_pool.enqueue(compute_overlaps, query_target_range, device_id));
    }

    // 3. Wait for all futures.
    for (auto& overlap_future : overlap_futures)
    {
        for (auto query_target_pair_overlaps : overlap_future.get())
        {
            query_target_pair_overlaps->wait();
        }
    }

    return 0;
}

void help(int32_t exit_code = 0)
{
    std::cerr <<
        R"(Usage: cudamapper [options ...] <query_sequences> <target_sequences>
     <sequences>
        Input file in FASTA/FASTQ format (can be compressed with gzip)
        containing sequences used for all-to-all overlapping
     options:
        -k, --kmer-size
            length of kmer to use for minimizers [15] (Max=)"
              << claragenomics::cudamapper::Index::maximum_kmer_size() << ")"
              << R"(
        -w, --window-size
            length of window to use for minimizers [15])"
              << R"(
        -d, --num-devices
            number of GPUs to use [1])"
              << R"(
        -c, --max_cache_size
            number of indices to keep in GPU memory [100])"
              << R"(
        -i, --index-size
            length of batch size used for query [10000])"
              << R"(
        -t --target-index-size
            length of batch sized used for target [10000])"
              << std::endl;

    exit(exit_code);
}
