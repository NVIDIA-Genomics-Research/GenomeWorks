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

/// @brief prints help message
/// @param exit_code
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
            Minimum ratio of overlap length to alignment length [0.95].)"
              << std::endl;

    exit(exit_code);
}

/// @brief application parameteres, default or passed through command line
struct ApplicationParameteres
{
    uint32_t k                                  = 15;   // k
    uint32_t w                                  = 15;   // w
    std::int32_t num_devices                    = 1;    // d
    std::int32_t max_index_cache_size_on_device = 100;  // c
    std::int32_t max_index_cache_size_on_host   = 0;    // C
    std::int32_t max_cached_memory              = 0;    // m
    std::int32_t index_size                     = 30;   // i
    std::int32_t target_index_size              = 30;   // t
    double filtering_parameter                  = 1.0;  // F
    std::int32_t alignment_engines              = 0;    // a
    std::int32_t min_residues                   = 10;   // r
    std::int32_t min_overlap_len                = 500;  // l
    std::int32_t min_bases_per_residue          = 100;  // b
    float min_overlap_fraction                  = 0.95; // z
    bool all_to_all                             = false;
    std::string query_filepath;
    std::string target_filepath;
};

/// @brief reads input from command line
/// @param argc
/// @param argv
/// @return application parameters passed through command line, default otherwise
ApplicationParameteres read_input(int argc, char* argv[])
{
    ApplicationParameteres parameters;

    struct option options[] = {
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

    std::string optstring = "k:w:d:c:C:m:i:t:F:h:a:r:l:b:z:";

    int32_t argument = 0;
    while ((argument = getopt_long(argc, argv, optstring.c_str(), options, nullptr)) != -1)
    {
        switch (argument)
        {
        case 'k':
            parameters.k = atoi(optarg);
            break;
        case 'w':
            parameters.w = atoi(optarg);
            break;
        case 'd':
            parameters.num_devices = atoi(optarg);
            break;
        case 'c':
            parameters.max_index_cache_size_on_device = atoi(optarg);
            break;
        case 'C':
            parameters.max_index_cache_size_on_host = atoi(optarg);
            break;
        case 'm':
#ifndef CGA_ENABLE_CACHING_ALLOCATOR
            std::cerr << "ERROR: Argument -m / --max-cached-memory cannot be used without caching allocator" << std::endl;
            exit(1);
#endif
            parameters.max_cached_memory = atoi(optarg);
            break;
        case 'i':
            parameters.index_size = atoi(optarg);
            break;
        case 't':
            parameters.target_index_size = atoi(optarg);
            break;
        case 'F':
            parameters.filtering_parameter = atof(optarg);
            break;
        case 'a':
            parameters.alignment_engines = atoi(optarg);
            claragenomics::throw_on_negative(parameters.alignment_engines, "Number of alignment engines should be non-negative");
            break;
        case 'r':
            parameters.min_residues = atoi(optarg);
            break;
        case 'l':
            parameters.min_overlap_len = atoi(optarg);
            break;
        case 'b':
            parameters.min_bases_per_residue = atoi(optarg);
            break;
        case 'z':
            parameters.min_overlap_fraction = atof(optarg);
            break;
        case 'h':
            help(0);
        default:
            exit(1);
        }
    }

    if (parameters.k > claragenomics::cudamapper::Index::maximum_kmer_size())
    {
        std::cerr << "kmer of size " << parameters.k << " is not allowed, maximum k = " << claragenomics::cudamapper::Index::maximum_kmer_size() << std::endl;
        exit(1);
    }

    if (parameters.filtering_parameter > 1.0 || parameters.filtering_parameter < 0.0)
    {
        std::cerr << "-F / --filtering-parameter must be in range [0.0, 1.0]" << std::endl;
        exit(1);
    }

    if (parameters.max_cached_memory < 0)
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

    parameters.query_filepath  = std::string(argv[optind++]);
    parameters.target_filepath = std::string(argv[optind++]);

    if (parameters.query_filepath == parameters.target_filepath)
    {
        parameters.all_to_all        = true;
        parameters.target_index_size = parameters.index_size;
        std::cerr << "NOTE - Since query and target files are same, activating all_to_all mode. Query index size used for both files." << std::endl;
    }

    return parameters;
}

/// @brief adds read names to overlaps and writes them to output
/// This function is expected to be executed async to matcher + overlapper
/// @param overlaps_writer_mtx locked while writing the output
/// @param num_overlap_chunks_to_print increased before the function is called, decreased right before the function finishes // TODO: improve this design
/// @param filtered_overlaps overlaps to be written out, on input without read names, on output cleared
/// @param query_index needed for read names // TODO: consider only passing vector of names, not whole indices
/// @param target_index needed for read names // TODO: consider only passing vector of names, not whole indices
/// @param cigar
/// @param device_id id of device on which query and target indices were created
void writer_thread_function(std::mutex& overlaps_writer_mtx,
                            std::atomic<int>& num_overlap_chunks_to_print,
                            std::shared_ptr<std::vector<claragenomics::cudamapper::Overlap>> filtered_overlaps,
                            std::shared_ptr<claragenomics::cudamapper::Index> query_index,
                            std::shared_ptr<claragenomics::cudamapper::Index> target_index,
                            const std::vector<std::string> cigar,
                            const int device_id,
                            const int kmer_size)
{
    // This function is expected to run in a separate thread so set current device in order to avoid problems
    // with deallocating indices with different current device than the one on which they were created
    cudaSetDevice(device_id);

    // Overlap post processing - add overlaps which can be combined into longer ones.
    claragenomics::cudamapper::Overlapper::post_process_overlaps(*filtered_overlaps);

    // parallel update of the query/target read names for filtered overlaps [parallel on host]
    claragenomics::cudamapper::Overlapper::update_read_names(*filtered_overlaps, *query_index, *target_index);
    std::lock_guard<std::mutex> lck(overlaps_writer_mtx);
    claragenomics::cudamapper::Overlapper::print_paf(*filtered_overlaps, cigar, kmer_size);

    //clear data
    for (auto o : *filtered_overlaps)
    {
        o.clear();
    }
    //Decrement counter which tracks number of overlap chunks to be filtered and printed
    num_overlap_chunks_to_print--;
};

int main(int argc, char* argv[])
{
    using claragenomics::get_size;
    claragenomics::logging::Init();

    const ApplicationParameteres parameters = read_input(argc, argv);

    std::shared_ptr<claragenomics::io::FastaParser> query_parser;
    std::shared_ptr<claragenomics::io::FastaParser> target_parser;

    query_parser = claragenomics::io::create_kseq_fasta_parser(parameters.query_filepath, parameters.k + parameters.w - 1);

    if (parameters.all_to_all)
    {
        target_parser = query_parser;
    }
    else
    {
        target_parser = claragenomics::io::create_kseq_fasta_parser(parameters.target_filepath, parameters.k + parameters.w - 1);
    }

    std::cerr << "Query file: " << parameters.query_filepath << ", number of reads: " << query_parser->get_num_seqences() << std::endl;
    std::cerr << "Target file: " << parameters.target_filepath << ", number of reads: " << target_parser->get_num_seqences() << std::endl;

    // Data structure for holding overlaps to be written out
    std::mutex overlaps_writer_mtx;

    struct QueryTargetsRange
    {
        std::pair<std::int32_t, int32_t> query_range;
        std::vector<std::pair<std::int32_t, int32_t>> target_ranges;
    };

    ///Factor of 1000000 to make max cache size in MB
    auto query_chunks  = query_parser->get_read_chunks(parameters.index_size * 1000000);
    auto target_chunks = target_parser->get_read_chunks(parameters.target_index_size * 1000000);

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
        if (parameters.all_to_all)
        {
            target_idx++;
        }
    }

    // This is host cache, if it has the index it will copy it to device, if not it will generate on device and add it to host cache
    std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<claragenomics::cudamapper::IndexHostCopyBase>> host_index_cache;

    // This is a per-device cache, if it has the index it will return it, if not it will generate it, store and return it.
    std::vector<std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<claragenomics::cudamapper::Index>>> device_index_cache(parameters.num_devices);

    // The number of overlap chunks which are to be computed
    std::atomic<int> num_overlap_chunks_to_print(0);

    auto get_index = [&device_index_cache, &host_index_cache, &parameters](claragenomics::DefaultDeviceAllocator allocator,
                                                                           claragenomics::io::FastaParser& parser,
                                                                           const claragenomics::read_id_t start_index,
                                                                           const claragenomics::read_id_t end_index,
                                                                           const std::uint64_t k,
                                                                           const std::uint64_t w,
                                                                           const int device_id,
                                                                           const bool allow_cache_index,
                                                                           const double filtering_parameter,
                                                                           const cudaStream_t cuda_stream) {
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
            index = host_index_cache[key]->copy_index_to_device(allocator,
                                                                cuda_stream);
        }
        else
        {
            //create an index, with hashed representations (minimizers)
            index = claragenomics::cudamapper::Index::create_index(allocator,
                                                                   parser,
                                                                   start_index,
                                                                   end_index,
                                                                   k,
                                                                   w,
                                                                   true, // hash_representations
                                                                   filtering_parameter,
                                                                   cuda_stream);

            // If in all-to-all mode, put this query in the cache for later use.
            // Cache eviction is handled later on by the calling thread
            // using the evict_index function.
            if (get_size<int32_t>(device_index_cache[device_id]) < parameters.max_index_cache_size_on_device && allow_cache_index)
            {
                device_index_cache[device_id][key] = index;
            }
            else if (get_size<int32_t>(host_index_cache) < parameters.max_index_cache_size_on_host && allow_cache_index && device_id == 0)
            {
                // if not cached on device, update host cache; only done on device 0 to avoid any race conditions in updating the host cache
                host_index_cache[key] = claragenomics::cudamapper::IndexHostCopyBase::create_cache(*index,
                                                                                                   start_index,
                                                                                                   k,
                                                                                                   w,
                                                                                                   cuda_stream);
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
    auto evict_index = [&device_index_cache, &host_index_cache](const claragenomics::read_id_t query_start_index,
                                                                const claragenomics::read_id_t query_end_index,
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
    if (parameters.max_cached_memory == 0)
    {
        std::cerr << "Programmatically looking for max cached memory" << std::endl;
        max_cached_bytes = claragenomics::cudautils::find_largest_contiguous_device_memory_section();
        if (max_cached_bytes == 0)
        {
            std::cerr << "No memory available for caching" << std::endl;
            exit(1);
        }
    }
    else
    {
        max_cached_bytes = parameters.max_cached_memory * 1024ull * 1024ull * 1024ull; // max_cached_memory is in GiB
    }

    std::cerr << "Using device memory cache of " << max_cached_bytes << " bytes" << std::endl;

    claragenomics::DefaultDeviceAllocator allocator(max_cached_bytes);
#else
    // uses CudaMallocAllocator
    claragenomics::DefaultDeviceAllocator allocator;
#endif

    auto compute_overlaps = [&](const QueryTargetsRange& query_target_range,
                                const int device_id,
                                const cudaStream_t cuda_stream) {
        auto query_start_index = query_target_range.query_range.first;
        auto query_end_index   = query_target_range.query_range.second;

        std::cerr << "Processing query range: (" << query_start_index << " - " << query_end_index - 1 << ")" << std::endl;

        std::shared_ptr<claragenomics::cudamapper::Index> query_index(nullptr);
        std::shared_ptr<claragenomics::cudamapper::Index> target_index(nullptr);
        std::unique_ptr<claragenomics::cudamapper::Matcher> matcher(nullptr);

        {
            CGA_NVTX_RANGE(profiler, "generate_query_index");
            query_index = get_index(allocator,
                                    *query_parser,
                                    query_start_index,
                                    query_end_index,
                                    parameters.k,
                                    parameters.w,
                                    device_id,
                                    parameters.all_to_all,
                                    parameters.filtering_parameter,
                                    cuda_stream);
        }

        //Main loop
        for (const auto target_range : query_target_range.target_ranges)
        {

            auto target_start_index = target_range.first;
            auto target_end_index   = target_range.second;
            {
                CGA_NVTX_RANGE(profiler, "generate_target_index");
                target_index = get_index(allocator,
                                         *target_parser,
                                         target_start_index,
                                         target_end_index,
                                         parameters.k,
                                         parameters.w,
                                         device_id,
                                         true,
                                         parameters.filtering_parameter,
                                         cuda_stream);
            }
            {
                CGA_NVTX_RANGE(profiler, "generate_matcher");
                matcher = claragenomics::cudamapper::Matcher::create_matcher(allocator,
                                                                             *query_index,
                                                                             *target_index,
                                                                             cuda_stream);
            }
            {

                claragenomics::cudamapper::OverlapperTriggered overlapper(allocator, cuda_stream);
                CGA_NVTX_RANGE(profiler, "generate_overlaps");

                // Get unfiltered overlaps
                auto overlaps_to_add = std::make_shared<std::vector<claragenomics::cudamapper::Overlap>>();

                overlapper.get_overlaps(*overlaps_to_add, matcher->anchors(),
                                        parameters.min_residues,
                                        parameters.min_overlap_len,
                                        parameters.min_bases_per_residue,
                                        parameters.min_overlap_fraction);

                std::vector<std::string> cigar;
                // Align overlaps
                if (parameters.alignment_engines > 0)
                {
                    cigar.resize(overlaps_to_add->size());
                    CGA_NVTX_RANGE(profiler, "align_overlaps");
                    claragenomics::cudamapper::Overlapper::Overlapper::align_overlaps(*overlaps_to_add, *query_parser, *target_parser, parameters.alignment_engines, cigar);
                }

                //Increment counter which tracks number of overlap chunks to be filtered and printed
                num_overlap_chunks_to_print++;

                std::thread t(writer_thread_function,
                              std::ref(overlaps_writer_mtx),
                              std::ref(num_overlap_chunks_to_print),
                              overlaps_to_add,
                              query_index,
                              target_index,
                              std::move(cigar),
                              device_id,
                              parameters.k);
                t.detach();
            }

            // reseting the matcher releases the anchor device array back to memory pool
            matcher.reset();
        }

        // If all-to-all mapping query will no longer be needed on device, remove it from the cache
        if (parameters.all_to_all)
        {
            evict_index(query_start_index, query_end_index, device_id, parameters.num_devices);
        }
    };

    // The application (File parsing, index generation, overlap generation etc) is all launched from here.
    // The main application works as follows:
    // 1. Launch a worker thread per device (GPU).
    // 2. Each worker takes target-query ranges off a queue
    // 3. Each worker pushes vector of futures (since overlap writing is dispatched to an async thread on host). All futures are waited for before the main application exits.
    std::vector<std::thread> workers;
    std::atomic<int> ranges_idx(0);

    // Each worker thread gets its own CUDA stream to work on. Currently there is only one worker thread per GPU,
    // but it is still necessary assign streams to each of then explicitly. --default-stream per-thread could
    // cause problems beacuse there are subthreads for worker threads
    std::vector<cudaStream_t> cuda_streams(parameters.num_devices);

    // Launch worker threads to enable multi-GPU.
    // One worker thread is responsible for one GPU so the number
    // of worker threads launched is equal to the number of devices specified
    // by the user
    for (int device_id = 0; device_id < parameters.num_devices; ++device_id)
    {
        CGA_CU_CHECK_ERR(cudaStreamCreate(&cuda_streams[device_id]));
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
                        compute_overlaps(query_target_ranges[range_idx],
                                         device_id,
                                         cuda_streams[device_id]);
                    }
                }
            }));
    }

    // Wait for all per-device threads to terminate
    for (auto& worker_thread : workers)
    {
        worker_thread.join();
    }

    // Wait for all futures (for overlap writing) to return
    while (num_overlap_chunks_to_print != 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // After last writer_thread_function has decreased num_overlap_chunks_to_print it will still take
    // some time to destroy its pointer to indices
    // TODO: this is a workaround, this part of code will be significantly changed with new index caching
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    device_index_cache.clear();

    // streams can only be destroyed once all writer threads have finished as they hold references
    // to indices which have device arrays associated with streams
    for (cudaStream_t cuda_stream : cuda_streams)
    {
        CGA_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
        CGA_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
    }

    return 0;
}
