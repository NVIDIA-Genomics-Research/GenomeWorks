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
#include <getopt.h>
#include <iostream>
#include <string>
#include <deque>
#include <mutex>
#include <future>
#include <thread>
#include <atomic>

#include <claragenomics/logging/logging.hpp>
#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/cudamapper/matcher.hpp>
#include <claragenomics/cudamapper/overlapper.hpp>
#include "overlapper_triggered.hpp"

static struct option options[] = {
    {"kmer-size", required_argument, 0, 'k'},
    {"window-size", required_argument, 0, 'w'},
    {"num-devices", required_argument, 0, 'd'},
    {"max-index-device-cache", required_argument, 0, 'c'},
    {"max-index-host-cache", required_argument, 0, 'C'},
    {"max-cached-memory", required_argument, 0, 'm'},
    {"index-size", required_argument, 0, 'i'},
    {"target-index-size", required_argument, 0, 't'},
    {"filtering-parameter", required_argument, 0, 'F'},
    {"alignment-engines", required_argument, 0, 'a'},
    {"min-residues", required_argument, 0, 'r'},
    {"min-overlap-length", required_argument, 0, 'l'},
    {"min-bases-per-residue", required_argument, 0, 'b'},
    {"min-overlap-fraction", required_argument, 0, 'z'},
    {"help", no_argument, 0, 'h'},
};

void help(int32_t exit_code);
std::size_t find_largest_contiguous_device_memory_section();

int main(int argc, char* argv[])
{
    using claragenomics::get_size;
    claragenomics::logging::Init();

    uint32_t k                                  = 15;  // k
    uint32_t w                                  = 15;  // w
    std::int32_t num_devices                    = 1;   // d
    std::int32_t max_index_cache_size_on_device = 100; // c
    // ToDo: come up with a good heuristic to choose C and c
    std::int32_t max_index_cache_size_on_host = 0;   // C
    std::int32_t max_cached_memory            = 0;   // m
    std::int32_t index_size                   = 30;  // i
    std::int32_t target_index_size            = 30;  // t
    double filtering_parameter                = 1.0; // F
    std::int32_t alignment_engines            = 0;   // a
    std::string optstring                     = "k:w:d:c:C:m:i:t:F:h:a:z:l:b:z:";
    int32_t argument                          = 0;
    std::int32_t min_residues                 = 20;
    std::int32_t min_overlap_len              = 200;
    std::int32_t min_bases_per_residue        = 50;
    float min_overlap_fraction                = 0.85;
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
            max_index_cache_size_on_device = atoi(optarg);
            break;
        case 'C':
            max_index_cache_size_on_host = atoi(optarg);
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
            claragenomics::throw_on_negative(alignment_engines, "Number of alignment engines should be non-negative");
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

    std::string query_filepath  = std::string(argv[optind++]);
    std::string target_filepath = std::string(argv[optind++]);

    bool all_to_all = false;
    if (query_filepath == target_filepath)
    {
        all_to_all        = true;
        target_index_size = index_size;
        std::cerr << "NOTE - Since query and target files are same, activating all_to_all mode. Query index size used for both files." << std::endl;
    }

    std::shared_ptr<claragenomics::io::FastaParser> query_parser = claragenomics::io::create_kseq_fasta_parser(query_filepath, k + w - 1);
    int32_t queries                                              = query_parser->get_num_seqences();

    std::shared_ptr<claragenomics::io::FastaParser> target_parser;
    if (all_to_all)
    {
        target_parser = query_parser;
    }
    else
    {
        target_parser = claragenomics::io::create_kseq_fasta_parser(target_filepath, k + w - 1);
    }

    int32_t targets = target_parser->get_num_seqences();

    std::cerr << "Query " << query_filepath << " index " << queries << std::endl;
    std::cerr << "Target " << target_filepath << " index " << targets << std::endl;

    // Data structure for holding overlaps to be written out
    std::mutex overlaps_writer_mtx;

    struct QueryTargetsRange
    {
        std::pair<std::int32_t, int32_t> query_range;
        std::vector<std::pair<std::int32_t, int32_t>> target_ranges;
    };

    ///Factor of 1000000 to make max cache size in MiB
    auto query_chunks  = query_parser->get_read_chunks(index_size * 1000000);
    auto target_chunks = target_parser->get_read_chunks(target_index_size * 1000000);

    //First generate all the ranges independently, then loop over them.
    std::vector<QueryTargetsRange> query_target_ranges;

    int target_idx = 0;
    for (auto const& query_chunk : query_chunks)
    {
        QueryTargetsRange range;
        range.query_range = query_chunk;
        for (size_t t = target_idx; t < target_chunks.size(); t++)
        {
            range.target_ranges.push_back(target_chunks[t]);
        }
        query_target_ranges.push_back(range);
        // in all-to-all, for query chunk 0, we go through target chunks [target_idx = 0 , n = target_chunks.size())
        // for query chunk 1, we only need target chunks [target_idx = 1 , n), and in general for query_chunk i, we need target chunks [target_idx = i , n)
        // therefore as we're looping through query chunks, in all-to-all, will increment target_idx
        if (all_to_all)
        {
            target_idx++;
        }
    }

    // This is host cache, if it has the index it will copy it to device, if not it will generate on device and add it to host cache
    std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<claragenomics::cudamapper::IndexHostCopy>> host_index_cache;

    // This is a per-device cache, if it has the index it will return it, if not it will generate it, store and return it.
    std::vector<std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<claragenomics::cudamapper::Index>>> device_index_cache(num_devices);

    // The number of overlap chunks which are to be computed
    std::atomic<int> num_overlap_chunks_to_print(0);

    auto get_index = [&device_index_cache, &host_index_cache, max_index_cache_size_on_device, max_index_cache_size_on_host](claragenomics::DefaultDeviceAllocator allocator,
                                                                                                                            claragenomics::io::FastaParser& parser,
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

        // first check if it's available on device, if not then check the host cache
        if (device_index_cache[device_id].count(key))
        {
            index = device_index_cache[device_id][key];
        }
        else if (host_index_cache.count(key))
        {
            index = host_index_cache[key]->copy_index_to_device(allocator);
        }
        else
        {
            //create an index, with hashed representations (minimizers)
            index = std::move(claragenomics::cudamapper::Index::create_index(allocator, parser, start_index, end_index, k, w, true, filtering_parameter));

            // If in all-to-all mode, put this query in the cache for later use.
            // Cache eviction is handled later on by the calling thread
            // using the evict_index function.
            if (get_size<int32_t>(device_index_cache[device_id]) < max_index_cache_size_on_device && allow_cache_index)
            {
                device_index_cache[device_id][key] = index;
            }
            else if (get_size<int32_t>(host_index_cache) < max_index_cache_size_on_host && allow_cache_index && device_id == 0)
            {
                // if not cached on device, update host cache; only done on device 0 to avoid any race conditions in updating the host cache
                host_index_cache[key] = std::move(claragenomics::cudamapper::IndexHostCopy::create_cache(*index, start_index, k, w));
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
    auto evict_index = [&device_index_cache, &host_index_cache](const claragenomics::cudamapper::read_id_t query_start_index,
                                                                const claragenomics::cudamapper::read_id_t query_end_index,
                                                                const int device_id,
                                                                const int num_devices) {
        std::pair<uint64_t, uint64_t> key;
        key.first  = query_start_index;
        key.second = query_end_index;
        device_index_cache[device_id].erase(key);
        // host memory can be freed by removing (key) when working with 1 GPU
        // in multiple GPUs we keep (key), as it may be accessed by other GPUs depending on access pattern
        if (num_devices == 1)
            host_index_cache.erase(key);
    };

#ifdef CGA_ENABLE_CACHING_ALLOCATOR
    // uses CachingDeviceAllocator
    std::size_t max_cached_bytes = 0;
    if (max_cached_memory == 0)
    {
        std::cerr << "Programmatically looking for max cached memory" << std::endl;
        max_cached_bytes = find_largest_contiguous_device_memory_section();
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

    claragenomics::DefaultDeviceAllocator allocator(max_cached_bytes);
#else
    // uses CudaMallocAllocator
    claragenomics::DefaultDeviceAllocator allocator;
#endif

    auto compute_overlaps = [&](const QueryTargetsRange& query_target_range, const int device_id) {
        auto query_start_index = query_target_range.query_range.first;
        auto query_end_index   = query_target_range.query_range.second;

        std::cerr << "Processing query range: (" << query_start_index << " - " << query_end_index - 1 << ")" << std::endl;

        std::shared_ptr<claragenomics::cudamapper::Index> query_index(nullptr);
        std::shared_ptr<claragenomics::cudamapper::Index> target_index(nullptr);
        std::unique_ptr<claragenomics::cudamapper::Matcher> matcher(nullptr);

        {
            CGA_NVTX_RANGE(profiler, "generate_query_index");
            query_index = get_index(allocator, *query_parser, query_start_index, query_end_index, k, w, device_id, all_to_all, filtering_parameter);
        }

        //Main loop
        for (const auto target_range : query_target_range.target_ranges)
        {

            auto target_start_index = target_range.first;
            auto target_end_index   = target_range.second;
            {
                CGA_NVTX_RANGE(profiler, "generate_target_index");
                target_index = get_index(allocator, *target_parser, target_start_index, target_end_index, k, w, device_id, true, filtering_parameter);
            }
            {
                CGA_NVTX_RANGE(profiler, "generate_matcher");
                matcher = claragenomics::cudamapper::Matcher::create_matcher(allocator, *query_index, *target_index);
            }
            {

                claragenomics::cudamapper::OverlapperTriggered overlapper(allocator);
                CGA_NVTX_RANGE(profiler, "generate_overlaps");

                // Get unfiltered overlaps
                auto overlaps_to_add = std::make_shared<std::vector<claragenomics::cudamapper::Overlap>>();

                overlapper.get_overlaps(*overlaps_to_add, matcher->anchors(), min_residues, min_overlap_len, min_bases_per_residue, min_overlap_fraction);

                std::vector<std::string> cigar;
                // Align overlaps
                if (alignment_engines > 0)
                {
                    cigar.resize(overlaps_to_add->size());
                    CGA_NVTX_RANGE(profiler, "align_overlaps");
                    claragenomics::cudamapper::Overlapper::Overlapper::align_overlaps(*overlaps_to_add, *query_parser, *target_parser, alignment_engines, cigar);
                }

                //Increment counter which tracks number of overlap chunks to be filtered and printed
                num_overlap_chunks_to_print++;
                auto print_overlaps = [&overlaps_writer_mtx, &num_overlap_chunks_to_print](std::shared_ptr<std::vector<claragenomics::cudamapper::Overlap>> filtered_overlaps,
                                                                                           std::shared_ptr<claragenomics::cudamapper::Index> query_index,
                                                                                           std::shared_ptr<claragenomics::cudamapper::Index> target_index,
                                                                                           const std::vector<std::string>& cigar,
                                                                                           const int device_id) {
                    // This lambda is expected to run in a separate thread so set current device in order to avoid problems
                    // with deallocating indices with different current device then the one on which they were created
                    cudaSetDevice(device_id);

                    // parallel update of the query/target read names for filtered overlaps [parallel on host]
                    claragenomics::cudamapper::Overlapper::update_read_names(*filtered_overlaps, *query_index, *target_index);
                    std::lock_guard<std::mutex> lck(overlaps_writer_mtx);
                    claragenomics::cudamapper::Overlapper::print_paf(*filtered_overlaps, cigar);

                    //clear data
                    for (auto o : *filtered_overlaps)
                    {
                        o.clear();
                    }
                    //Decrement counter which tracks number of overlap chunks to be filtered and printed
                    num_overlap_chunks_to_print--;
                };

                std::thread t(print_overlaps, overlaps_to_add, query_index, target_index, cigar, device_id);
                t.detach();
            }

            // reseting the matcher releases the anchor device array back to memory pool
            matcher.reset();
        }

        // If all-to-all mapping query will no longer be needed on device, remove it from the cache
        if (all_to_all)
        {
            evict_index(query_start_index, query_end_index, device_id, num_devices);
        }
    };

    // The application (File parsing, index generation, overlap generation etc) is all launched from here.
    // The main application works as follows:
    // 1. Launch a worker thread per device (GPU).
    // 2. Each worker takes target-query ranges off a queue
    // 3. Each worker pushes vector of futures (since overlap writing is dispatched to an async thread on host). All futures are waited for before the main application exits.
    std::vector<std::thread> workers;
    std::atomic<int> ranges_idx(0);

    // Launch worker threads to enable multi-GPU.
    // One worker thread is responsible for one GPU so the number
    // of worker threads launched is equal to the number of devices specified
    // by the user
    for (int device_id = 0; device_id < num_devices; device_id++)
    {
        //Worker thread consumes query-target ranges off a queue
        workers.push_back(std::thread(
            [&, device_id]() {
                cudaSetDevice(device_id);
                while (ranges_idx < get_size<int>(query_target_ranges))
                {
                    int range_idx = ranges_idx.fetch_add(1);
                    //Need to perform this check again for thread-safety
                    if (range_idx < get_size<int>(query_target_ranges))
                    {
                        //compute overlaps takes a range of read_ids and a device ID and uses
                        //that device to compute the overlaps. It prints overlaps to stdout.
                        //since multiple worker threads are running stdout is guarded
                        //by a mutex (`std::mutex overlaps_writer_mtx`)
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
    while (num_overlap_chunks_to_print != 0)
    {
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
        -c, --max-index-device-cache
            number of indices to keep in GPU memory [100])"
              << R"(
        -C, --max-index-host-cache
            number of indices to keep in host memory [0])"
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
              << std::endl;

    exit(exit_code);
}

/// @brief finds largest section of contiguous memory on device
/// @return number of bytes
std::size_t find_largest_contiguous_device_memory_section()
{
    // find the largest block of contiguous memory
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    const size_t memory_decrement = free / 100;              // decrease requested memory one by one percent
    size_t size_to_try            = free - memory_decrement; // do not go for all memory
    while (true)
    {
        void* dummy_ptr    = nullptr;
        cudaError_t status = cudaMalloc(&dummy_ptr, size_to_try);
        // if it was able to allocate memory free the memory and return the size
        if (status == cudaSuccess)
        {
            cudaFree(dummy_ptr);
            return size_to_try;
        }

        if (status == cudaErrorMemoryAllocation)
        {
            // if it was not possible to allocate the memory because there was not enough of it
            // try allocating less memory in next iteration
            if (size_to_try > memory_decrement)
            {
                size_to_try -= memory_decrement;
            }
            else
            { // a very small amount of memory left, report an error
                CGA_CU_CHECK_ERR(cudaErrorMemoryAllocation);
                return 0;
            }
        }
        else
        {
            // if cudaMalloc failed because of error other than cudaErrorMemoryAllocation process the error
            CGA_CU_CHECK_ERR(status);
        }
    }

    // this point should actually never be reached (loop either finds memory or causes an error)
    assert(false);
    CGA_CU_CHECK_ERR(cudaErrorMemoryAllocation);
    return 0;
}