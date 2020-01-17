/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
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

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <claragenomics/logging/logging.hpp>
#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/cudautils.hpp>

#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/cudamapper/matcher.hpp>
#include <claragenomics/cudamapper/overlapper.hpp>
#include "overlapper_triggered.hpp"
#include "../../common/io/src/kseqpp_fasta_parser.hpp"

static struct option options[] = {
    {"kmer-size", required_argument, 0, 'k'},
    {"window-size", required_argument, 0, 'w'},
    {"num-devices", required_argument, 0, 'd'},
    {"max-cache-size", required_argument, 0, 'c'},
    {"index-size", required_argument, 0, 'i'},
    {"target-index-size", required_argument, 0, 't'},
    {"filtering-parameter", required_argument, 0, 'F'},
    {"help", no_argument, 0, 'h'},
};

void help(int32_t exit_code);

int main(int argc, char* argv[])
{
    claragenomics::logging::Init();

    uint32_t k                     = 15;    // k
    uint32_t w                     = 15;    // w
    std::int32_t num_devices       = 1;     // d
    std::int32_t max_cache_size    = 100;   // c
    std::int32_t index_size        = 10000; // i
    std::int32_t target_index_size = 10000; // t
    double filtering_parameter     = 1.0;   // F
    std::string optstring          = "k:w:d:c:i:t:F:h:";
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
        case 'd':
            num_devices = atoi(optarg);
            break;
        case 'c':
            max_cache_size = atoi(optarg);
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

    if (filtering_parameter > 1.0 || filtering_parameter < 0.0)
    {
        std::cerr << "-F / --filtering-parameter must be in range [0.0, 1.0]" << std::endl;
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


    auto parsers = claragenomics::io::create_kseq_fasta_parser(query_filepath);

    std::shared_ptr<claragenomics::io::FastaParser> query_parser = claragenomics::io::create_kseq_fasta_parser(query_filepath);
    int32_t queries                                              = query_parser->get_num_seqences();

    std::shared_ptr<claragenomics::io::FastaParser> target_parser;
    if (all_to_all){
        target_parser = query_parser;
    } else {

    }
    target_parser = claragenomics::io::create_kseq_fasta_parser(target_filepath);

    int32_t targets                                               = target_parser->get_num_seqences();

    std::cerr << "Query " << query_filepath << " index " << queries << std::endl;
    std::cerr << "Target " << target_filepath << " index " << targets << std::endl;

    // Data structure for holding overlaps to be written out
    std::mutex overlaps_writer_mtx;

    struct query_target_range
    {
        std::pair<std::int32_t, int32_t> query_range;
        std::vector<std::pair<std::int32_t, int32_t>> target_ranges;
    };

    auto query_chunks = query_parser->get_read_chunks(index_size * 1000000);
    auto target_chunks = target_parser->get_read_chunks(target_index_size * 1000000);

    //First generate all the ranges independently, then loop over them.
    std::vector<query_target_range> query_target_ranges;


    // TODO: here goes filling out the query_target_ranges vector but using chunks

    int target_idx = 0;
    for (auto query_chunk: query_chunks){
        query_target_range range;
        range.query_range = query_chunk;
        for (int t = target_idx; t<target_chunks.size(); t++){
            range.target_ranges.push_back(target_chunks[t]);
        }
        query_target_ranges.push_back(range);
        if(all_to_all){
            target_idx++;
        }
    }


/*    for (std::int32_t query_start_index = 0; query_start_index < queries; query_start_index += index_size)
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
    }*/

    // This is a per-device cache, if it has the index it will return it, if not it will generate it, store and return it.
    std::vector<std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<claragenomics::cudamapper::Index>>> index_cache(num_devices);

    std::atomic<int> num_overlap_chunks(0);

    auto get_index = [&index_cache, max_cache_size](claragenomics::io::FastaParser& parser,
                                                    const claragenomics::cudamapper::read_id_t start_index,
                                                    const claragenomics::cudamapper::read_id_t end_index,
                                                    const std::uint64_t k,
                                                    const std::uint64_t w,
                                                    const int device_id,
                                                    const bool allow_cache_index,
                                                    const double filtering_parameter) {
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
            //std::cerr<< "Using filtering aram of" << filtering_parameter << std::endl;
            index = std::move(claragenomics::cudamapper::Index::create_index(parser, start_index, end_index, k, w, true, filtering_parameter));

            // If in all-to-all mode, put this query in the cache for later use.
            // Cache eviction is handled later on by the calling thread
            // using the evict_index function.
            if (index_cache[device_id].size() < max_cache_size && allow_cache_index)
            {
                index_cache[device_id][key] = index;
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
    auto evict_index = [&index_cache](const claragenomics::cudamapper::read_id_t query_start_index,
                                      const claragenomics::cudamapper::read_id_t query_end_index,
                                      const int device_id) {
        std::pair<uint64_t, uint64_t> key;
        key.first  = query_start_index;
        key.second = query_end_index;
        index_cache[device_id].erase(key);
    };

    auto compute_overlaps = [&](const query_target_range query_target_range, const int device_id) {
        std::vector<std::shared_ptr<std::future<void>>> print_pafs_futures;

        cudaSetDevice(device_id);

        auto query_start_index = query_target_range.query_range.first;
        auto query_end_index   = query_target_range.query_range.second;

        std::cerr << "Processing query range: (" << query_start_index << " - " << query_end_index << ")" << std::endl;

        std::shared_ptr<claragenomics::cudamapper::Index> query_index(nullptr);
        std::shared_ptr<claragenomics::cudamapper::Index> target_index(nullptr);
        std::unique_ptr<claragenomics::cudamapper::Matcher> matcher(nullptr);

        {
            CGA_NVTX_RANGE(profiler, "generate_query_index");
            query_index = get_index(*query_parser, query_start_index, query_end_index, k, w, device_id, all_to_all, filtering_parameter);
        }

        //Main loop
        for (const auto target_range : query_target_range.target_ranges)
        {

            auto target_start_index = target_range.first;
            auto target_end_index   = target_range.second;
            {
                CGA_NVTX_RANGE(profiler, "generate_target_index");
                target_index = get_index(*target_parser, target_start_index, target_end_index, k, w, device_id, true, filtering_parameter);
            }
            {
                CGA_NVTX_RANGE(profiler, "generate_matcher");
                matcher = claragenomics::cudamapper::Matcher::create_matcher(*query_index,
                                                                             *target_index);
            }
            {

                claragenomics::cudamapper::OverlapperTriggered overlapper;
                CGA_NVTX_RANGE(profiler, "generate_overlaps");

                // Get unfiltered overlaps
                std::vector<claragenomics::cudamapper::Overlap, thrust::system::cuda::experimental::pinned_allocator<claragenomics::cudamapper::Overlap>> overlaps_to_add;

                overlapper.get_overlaps(overlaps_to_add, matcher->anchors(), *query_index, *target_index);

                num_overlap_chunks += 1;
                std::async(std::launch::async,
                         [&overlaps_writer_mtx, &overlaps_to_add, &num_overlap_chunks](std::vector<claragenomics::cudamapper::Overlap, thrust::system::cuda::experimental::pinned_allocator<claragenomics::cudamapper::Overlap>> overlaps) {
                             std::vector<claragenomics::cudamapper::Overlap, thrust::system::cuda::experimental::pinned_allocator<claragenomics::cudamapper::Overlap>> filtered_overlaps;
                             claragenomics::cudamapper::Overlapper::filter_overlaps(filtered_overlaps, overlaps, 50);
                             std::lock_guard<std::mutex> lck(overlaps_writer_mtx);
                             claragenomics::cudamapper::Overlapper::print_paf(filtered_overlaps);

                             //clear data
                             for (auto o: overlaps){
                                 o.clear();
                             }

                             //
                             num_overlap_chunks--;

                             }, overlaps_to_add);
            }
        }

        // If all-to-all mapping query will no longer be needed on device, remove it from the cache
        if (all_to_all)
        {
            evict_index(query_start_index, query_end_index, device_id);
        }

        return print_pafs_futures;
    };

    // The application (File parsing, index generation, overlap generation etc) is all launched from here.
    // The main application works as follows:
    // 1. Launch a worker thread per device (GPU).
    // 2. Each worker takes target-query ranges off a queue
    // 3. Each worker pushes vector of futures (since overlap writing is dispatched to an async thread on host). All futures are waited for before the main application exits.
    std::vector<std::thread> workers;
    std::atomic<int> ranges_idx(0);

    // Launch worker threads
    for (int device_id = 0; device_id < num_devices; device_id++)
    {
        std::cerr << "Launching worker thread" << std::endl;
        //Worker thread consumes query-target ranges off a queue
        workers.push_back(std::thread(
                [&, device_id]() {
                    while (ranges_idx < query_target_ranges.size()) {
                        int range_idx = ranges_idx.fetch_add(1);
                        //Need to perform this check again for thread-safety
                        if (range_idx < query_target_ranges.size()) {
                            compute_overlaps(query_target_ranges[range_idx], device_id);
                        }
                    }
                }));
    }

    // Wait for all per-device threads to terminate
    std::for_each(workers.begin(), workers.end(), [](std::thread& t) {
        t.join();
    });

    // Wait for all futures (for overlap filtering and writing) to return
    while(num_overlap_chunks !=0){
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
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
              << R"(
        -F --filtering-parameter
            filter all representations for which sketch_elements_with_that_representation/total_sketch_elements >= filtering_parameter), filtering disabled if filtering_parameter == 1.0 [1'000'000'001] (Min = 0.0, Max = 1.0))"
              << std::endl;

    exit(exit_code);
}
