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
#include <map>
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
#include "index_descriptor.hpp"

#include <claragenomics/cudaaligner/aligner.hpp>
#include <claragenomics/cudaaligner/alignment.hpp>

namespace claragenomics
{
namespace cudamapper
{

namespace
{

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
              << Index::maximum_kmer_size() << ")"
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
            throw_on_negative(parameters.alignment_engines, "Number of alignment engines should be non-negative");
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

    if (parameters.k > Index::maximum_kmer_size())
    {
        std::cerr << "kmer of size " << parameters.k << " is not allowed, maximum k = " << Index::maximum_kmer_size() << std::endl;
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

void run_alignment_batch(DefaultDeviceAllocator allocator,
                         std::mutex& overlap_idx_mtx,
                         std::vector<Overlap>& overlaps,
                         const io::FastaParser& query_parser,
                         const io::FastaParser& target_parser,
                         int32_t& overlap_idx,
                         const int32_t max_query_size, const int32_t max_target_size,
                         std::vector<std::string>& cigar, const int32_t batch_size)
{
    int32_t device_id;
    CGA_CU_CHECK_ERR(cudaGetDevice(&device_id));
    cudaStream_t stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&stream));
    std::unique_ptr<cudaaligner::Aligner> batch =
        cudaaligner::create_aligner(
            max_query_size,
            max_target_size,
            batch_size,
            cudaaligner::AlignmentType::global_alignment,
            allocator,
            stream,
            device_id);
    while (true)
    {
        int32_t idx_start = 0, idx_end = 0;
        // Get the range of overlaps for this batch
        {
            std::lock_guard<std::mutex> lck(overlap_idx_mtx);
            if (overlap_idx == get_size<int32_t>(overlaps))
            {
                break;
            }
            else
            {
                idx_start   = overlap_idx;
                idx_end     = std::min(idx_start + batch_size, get_size<int32_t>(overlaps));
                overlap_idx = idx_end;
            }
        }
        for (int32_t idx = idx_start; idx < idx_end; idx++)
        {
            const Overlap& overlap         = overlaps[idx];
            const io::FastaSequence query  = query_parser.get_sequence_by_id(overlap.query_read_id_);
            const io::FastaSequence target = target_parser.get_sequence_by_id(overlap.target_read_id_);
            const char* query_start        = &query.seq[overlap.query_start_position_in_read_];
            const int32_t query_length     = overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_;
            const char* target_start       = &target.seq[overlap.target_start_position_in_read_];
            const int32_t target_length    = overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_;
            cudaaligner::StatusType status = batch->add_alignment(query_start, query_length, target_start, target_length,
                                                                  false, overlap.relative_strand == RelativeStrand::Reverse);
            if (status != cudaaligner::success)
            {
                throw std::runtime_error("Experienced error type " + std::to_string(status));
            }
        }
        // Launch alignment on the GPU. align_all is an async call.
        batch->align_all();
        // Synchronize all alignments.
        batch->sync_alignments();
        const std::vector<std::shared_ptr<cudaaligner::Alignment>>& alignments = batch->get_alignments();
        {
            CGA_NVTX_RANGE(profiler, "copy_alignments");
            for (int32_t i = 0; i < get_size<int32_t>(alignments); i++)
            {
                cigar[idx_start + i] = alignments[i]->convert_to_cigar();
            }
        }
        // Reset batch to reuse memory for new alignments.
        batch->reset();
    }
    CGA_CU_CHECK_ERR(cudaStreamDestroy(stream));
}

/// \brief performs gloval alignment between overlapped regions of reads
/// \param overlaps List of overlaps to align
/// \param query_parser Parser for query reads
/// \param target_parser Parser for target reads
/// \param num_alignment_engines Number of parallel alignment engines to use for alignment
/// \param cigar Output vector to store CIGAR string for alignments
/// \param allocator The allocator to allocate memory on the device
void align_overlaps(DefaultDeviceAllocator allocator,
                    std::vector<Overlap>& overlaps,
                    const io::FastaParser& query_parser,
                    const io::FastaParser& target_parser,
                    int32_t num_alignment_engines,
                    std::vector<std::string>& cigar)
{
    // Calculate max target/query size in overlaps
    int32_t max_query_size  = 0;
    int32_t max_target_size = 0;
    for (const auto& overlap : overlaps)
    {
        int32_t query_overlap_size  = overlap.query_end_position_in_read_ - overlap.query_start_position_in_read_;
        int32_t target_overlap_size = overlap.target_end_position_in_read_ - overlap.target_start_position_in_read_;
        if (query_overlap_size > max_query_size)
            max_query_size = query_overlap_size;
        if (target_overlap_size > max_target_size)
            max_target_size = target_overlap_size;
    }

    // Heuristically calculate max alignments possible with available memory based on
    // empirical measurements of memory needed for alignment per base.
    const float memory_per_base = 0.03f; // Estimation of space per base in bytes for alignment
    float memory_per_alignment  = memory_per_base * max_query_size * max_target_size;
    size_t free, total;
    CGA_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
    const size_t max_alignments = (static_cast<float>(free) * 85 / 100) / memory_per_alignment; // Using 85% of available memory
    int32_t batch_size          = std::min(get_size<int32_t>(overlaps), static_cast<int32_t>(max_alignments)) / num_alignment_engines;
    std::cerr << "Aligning " << overlaps.size() << " overlaps (" << max_query_size << "x" << max_target_size << ") with batch size " << batch_size << std::endl;

    int32_t overlap_idx = 0;
    std::mutex overlap_idx_mtx;

    // Launch multiple alignment engines in separate threads to overlap D2H and H2D copies
    // with compute from concurrent engines.
    std::vector<std::future<void>> align_futures;
    for (int32_t t = 0; t < num_alignment_engines; t++)
    {
        align_futures.push_back(std::async(std::launch::async,
                                           &run_alignment_batch,
                                           allocator,
                                           std::ref(overlap_idx_mtx),
                                           std::ref(overlaps),
                                           std::ref(query_parser),
                                           std::ref(target_parser),
                                           std::ref(overlap_idx),
                                           max_query_size,
                                           max_target_size,
                                           std::ref(cigar),
                                           batch_size));
    }

    for (auto& f : align_futures)
    {
        f.get();
    }
}

/// @brief adds read names to overlaps and writes them to output
/// This function is expected to be executed async to matcher + overlapper
/// @param overlaps_writer_mtx locked while writing the output
/// @param num_overlap_chunks_to_print increased before the function is called, decreased right before the function finishes // TODO: improve this design
/// @param filtered_overlaps overlaps to be written out, on input without read names, on output cleared
/// @param query_parser needed for read names and lenghts
/// @param target_parser needed for read names and lenghts
/// @param cigar
/// @param device_id id of device on which query and target indices were created
void writer_thread_function(std::mutex& overlaps_writer_mtx,
                            std::atomic<int>& num_overlap_chunks_to_print,
                            std::shared_ptr<std::vector<Overlap>> filtered_overlaps,
                            const io::FastaParser& query_parser,
                            const io::FastaParser& target_parser,
                            const std::vector<std::string> cigar,
                            const int device_id,
                            const int kmer_size)
{
    // This function is expected to run in a separate thread so set current device in order to avoid problems
    // with deallocating indices with different current device than the one on which they were created
    cudaSetDevice(device_id);

    // Overlap post processing - add overlaps which can be combined into longer ones.
    Overlapper::post_process_overlaps(*filtered_overlaps);

    // parallel update of the query/target read names for filtered overlaps [parallel on host]
    Overlapper::update_read_names(*filtered_overlaps, query_parser, target_parser);
    std::lock_guard<std::mutex> lck(overlaps_writer_mtx);
    Overlapper::print_paf(*filtered_overlaps, cigar, kmer_size);

    //clear data
    for (auto o : *filtered_overlaps)
    {
        o.clear();
    }
    //Decrement counter which tracks number of overlap chunks to be filtered and printed
    num_overlap_chunks_to_print--;
};

} // namespace

int main(int argc, char* argv[])
{
    logging::Init();

    const ApplicationParameteres parameters = read_input(argc, argv);

    std::shared_ptr<io::FastaParser> query_parser;
    std::shared_ptr<io::FastaParser> target_parser;

    query_parser = io::create_kseq_fasta_parser(parameters.query_filepath, parameters.k + parameters.w - 1);

    if (parameters.all_to_all)
    {
        target_parser = query_parser;
    }
    else
    {
        target_parser = io::create_kseq_fasta_parser(parameters.target_filepath, parameters.k + parameters.w - 1);
    }

    std::cerr << "Query file: " << parameters.query_filepath << ", number of reads: " << query_parser->get_num_seqences() << std::endl;
    std::cerr << "Target file: " << parameters.target_filepath << ", number of reads: " << target_parser->get_num_seqences() << std::endl;

    // Data structure for holding overlaps to be written out
    std::mutex overlaps_writer_mtx;

    struct QueryTargetsRange
    {
        IndexDescriptor query_range;
        std::vector<IndexDescriptor> target_ranges;
    };

    ///Factor of 1000000 to make max cache size in MB
    std::vector<IndexDescriptor> query_index_descriptors  = group_reads_into_indices(*query_parser,
                                                                                    parameters.index_size * 1000000);
    std::vector<IndexDescriptor> target_index_descriptors = group_reads_into_indices(*target_parser,
                                                                                     parameters.target_index_size * 1000000);

    //First generate all the ranges independently, then loop over them.
    std::vector<QueryTargetsRange> query_target_ranges;

    int target_idx = 0;
    for (const IndexDescriptor& query_index_descriptor : query_index_descriptors)
    {
        QueryTargetsRange range{query_index_descriptor, {}};
        for (size_t t = target_idx; t < target_index_descriptors.size(); t++)
        {
            range.target_ranges.push_back(target_index_descriptors[t]);
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
    std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<IndexHostCopyBase>> host_index_cache;

    // This is a per-device cache, if it has the index it will return it, if not it will generate it, store and return it.
    std::vector<std::map<std::pair<uint64_t, uint64_t>, std::shared_ptr<Index>>> device_index_cache(parameters.num_devices);

    // The number of overlap chunks which are to be computed
    std::atomic<int> num_overlap_chunks_to_print(0);

    auto get_index = [&device_index_cache, &host_index_cache, &parameters](DefaultDeviceAllocator allocator,
                                                                           io::FastaParser& parser,
                                                                           const read_id_t start_index,
                                                                           const read_id_t end_index,
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

        std::shared_ptr<Index> index;

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
            index = Index::create_index(allocator,
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
                host_index_cache[key] = IndexHostCopyBase::create_cache(*index,
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
    auto evict_index = [&device_index_cache, &host_index_cache](const read_id_t query_start_index,
                                                                const read_id_t query_end_index,
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
        max_cached_bytes = cudautils::find_largest_contiguous_device_memory_section();
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

    DefaultDeviceAllocator allocator(max_cached_bytes);
#else
    // uses CudaMallocAllocator
    DefaultDeviceAllocator allocator;
#endif

    auto compute_overlaps = [&](const QueryTargetsRange& query_target_range,
                                const int device_id,
                                const cudaStream_t cuda_stream) {
        auto query_start_index = query_target_range.query_range.first_read();
        auto query_end_index   = query_target_range.query_range.first_read() + query_target_range.query_range.number_of_reads();

        std::cerr << "Processing query range: (" << query_start_index << " - " << query_end_index - 1 << ")" << std::endl;

        std::shared_ptr<Index> query_index(nullptr);
        std::shared_ptr<Index> target_index(nullptr);
        std::unique_ptr<Matcher> matcher(nullptr);

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
        for (const IndexDescriptor& target_range : query_target_range.target_ranges)
        {

            auto target_start_index = target_range.first_read();
            auto target_end_index   = target_range.first_read() + target_range.number_of_reads();
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
                matcher = Matcher::create_matcher(allocator,
                                                  *query_index,
                                                  *target_index,
                                                  cuda_stream);
            }
            {

                OverlapperTriggered overlapper(allocator, cuda_stream);
                CGA_NVTX_RANGE(profiler, "generate_overlaps");

                // Get unfiltered overlaps
                auto overlaps_to_add = std::make_shared<std::vector<Overlap>>();

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
                    align_overlaps(allocator, *overlaps_to_add, *query_parser, *target_parser, parameters.alignment_engines, cigar);
                }

                //Increment counter which tracks number of overlap chunks to be filtered and printed
                num_overlap_chunks_to_print++;

                std::thread t(writer_thread_function,
                              std::ref(overlaps_writer_mtx),
                              std::ref(num_overlap_chunks_to_print),
                              overlaps_to_add,
                              std::ref(*query_parser),
                              std::ref(*target_parser),
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

} // namespace cudamapper
} // namespace claragenomics

/// \brief main function
/// main function cannot be in a namespace so using this function to call actual main function
int main(int argc, char* argv[])
{
    return claragenomics::cudamapper::main(argc, argv);
}
