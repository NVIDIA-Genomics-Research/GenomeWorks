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

#include <atomic>
#include <algorithm>
#include <iostream>
#include <future>
#include <mutex>
#include <string>
#include <thread>

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/mathutils.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/utils/threadsafe_containers.hpp>

#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>
#include <claraparabricks/genomeworks/cudaaligner/alignment.hpp>

#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/cudamapper/matcher.hpp>
#include <claraparabricks/genomeworks/cudamapper/overlapper.hpp>
#include <claraparabricks/genomeworks/cudamapper/utils.hpp>

#include "application_parameters.hpp"
#include "cudamapper_utils.hpp"
#include "index_batcher.cuh"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

namespace
{

void run_alignment_batch(DefaultDeviceAllocator allocator,
                         std::mutex& overlap_idx_mtx,
                         std::vector<Overlap>& overlaps,
                         const io::FastaParser& query_parser,
                         const io::FastaParser& target_parser,
                         int32_t& overlap_idx,
                         const int32_t max_query_size, const int32_t max_target_size,
                         std::vector<std::string>& cigars, const int32_t batch_size)
{
    int32_t device_id;
    GW_CU_CHECK_ERR(cudaGetDevice(&device_id));
    CudaStream stream = make_cuda_stream();
    std::unique_ptr<cudaaligner::Aligner> batch =
        cudaaligner::create_aligner(
            max_query_size,
            max_target_size,
            batch_size,
            cudaaligner::AlignmentType::global_alignment,
            allocator,
            stream.get(),
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
            GW_NVTX_RANGE(profiler, "copy_alignments");
            for (int32_t i = 0; i < get_size<int32_t>(alignments); i++)
            {
                cigars[idx_start + i] = alignments[i]->convert_to_cigar();
            }
        }
        // Reset batch to reuse memory for new alignments.
        batch->reset();
    }
}

/// \brief performs global alignment between overlapped regions of reads
/// \param overlaps List of overlaps to align
/// \param query_parser Parser for query reads
/// \param target_parser Parser for target reads
/// \param num_alignment_engines Number of parallel alignment engines to use for alignment
/// \param cigars Output vector to store CIGAR strings for alignments
/// \param allocator The allocator to allocate memory on the device
void align_overlaps(DefaultDeviceAllocator allocator,
                    std::vector<Overlap>& overlaps,
                    const io::FastaParser& query_parser,
                    const io::FastaParser& target_parser,
                    int32_t num_alignment_engines,
                    std::vector<std::string>& cigars)
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
    GW_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
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
                                           std::ref(cigars),
                                           batch_size));
    }

    for (auto& f : align_futures)
    {
        f.get();
    }
}

/// OverlapsAndCigars - packs overlaps and cigars together so they can be passed to writer thread more easily
struct OverlapsAndCigars
{
    std::vector<Overlap> overlaps;
    std::vector<std::string> cigars;
};

/// \brief does overlapping and matching for pairs of query and target indices from device_batch
/// \param device_batch
/// \param device_cache data will be loaded into cache within the function
/// \param application_parameters
/// \param overlaps_and_cigars_to_process overlaps and cigars are output here and the then consumed by another thread
/// \param number_of_skipped_pairs_of_indices number of pairs of indices skipped due to OOM error, variable shared between all threads, each call increases the number by the number of skipped pairs
/// \param cuda_stream
void process_one_device_batch(const IndexBatch& device_batch,
                              IndexCacheDevice& device_cache,
                              const ApplicationParameters& application_parameters,
                              DefaultDeviceAllocator device_allocator,
                              ThreadsafeProducerConsumer<OverlapsAndCigars>& overlaps_and_cigars_to_process,
                              std::atomic<int32_t>& number_of_skipped_pairs_of_indices,
                              cudaStream_t cuda_stream)
{
    GW_NVTX_RANGE(profiler, "main::process_one_device_batch");
    const std::vector<IndexDescriptor>& query_index_descriptors  = device_batch.query_indices;
    const std::vector<IndexDescriptor>& target_index_descriptors = device_batch.target_indices;

    // fetch indices for this batch from host memory
    assert(!query_index_descriptors.empty() && !target_index_descriptors.empty());
    device_cache.generate_query_cache_content(query_index_descriptors);
    device_cache.generate_target_cache_content(target_index_descriptors);

    // process pairs of query and target indices
    for (const IndexDescriptor& query_index_descriptor : query_index_descriptors)
    {
        for (const IndexDescriptor& target_index_descriptor : target_index_descriptors)
        {
            // if doing all-to-all skip pairs in which target batch has smaller id than query batch as it will be covered by symmetry
            if (!application_parameters.all_to_all || target_index_descriptor.first_read() >= query_index_descriptor.first_read())
            {
                std::shared_ptr<Index> query_index  = device_cache.get_index_from_query_cache(query_index_descriptor);
                std::shared_ptr<Index> target_index = device_cache.get_index_from_target_cache(target_index_descriptor);

                try
                {
                    // find anchors and overlaps
                    auto matcher = Matcher::create_matcher(device_allocator,
                                                           *query_index,
                                                           *target_index,
                                                           cuda_stream);

                    std::vector<Overlap> overlaps;
                    auto overlapper = Overlapper::create_overlapper(device_allocator,
                                                                    cuda_stream);

                    overlapper->get_overlaps(overlaps,
                                             matcher->anchors(),
                                             application_parameters.all_to_all,
                                             application_parameters.min_residues,
                                             application_parameters.min_overlap_len,
                                             application_parameters.min_bases_per_residue,
                                             application_parameters.min_overlap_fraction);

                    // free up memory taken by matcher
                    matcher.reset(nullptr);

                    // Align overlaps
                    std::vector<std::string> cigars;
                    if (application_parameters.alignment_engines > 0)
                    {
                        cigars.resize(overlaps.size());
                        GW_NVTX_RANGE(profiler, "align_overlaps");
                        align_overlaps(device_allocator,
                                       overlaps,
                                       *application_parameters.query_parser,
                                       *application_parameters.target_parser,
                                       application_parameters.alignment_engines,
                                       cigars);
                    }

                    // pass overlaps and cigars to writer thread
                    overlaps_and_cigars_to_process.add_new_element({std::move(overlaps), std::move(cigars)});
                }
                catch (device_memory_allocation_exception& oom_exception)
                {
                    // if the application ran out of memory skip this pair of indices
                    ++(number_of_skipped_pairs_of_indices);
                }
            }
        }
    }
}

/// \brief loads one batch into host memory and then processes its device batches one by one
/// \param batch
/// \param application_parameters
/// \param host_cache data will be loaded into cache within the function
/// \param device_cache data will be loaded into cache within the function
/// \param overlaps_and_cigars_to_process overlaps and cigars are output to this structure and the then consumed by another thread
/// \param number_of_skipped_pairs_of_indices number of pairs of indices skipped due to OOM error, variable shared between all threads, each call increases the number by the number of skipped pairs
/// \param cuda_stream
void process_one_batch(const BatchOfIndices& batch,
                       const ApplicationParameters& application_parameters,
                       DefaultDeviceAllocator device_allocator,
                       IndexCacheHost& host_cache,
                       IndexCacheDevice& device_cache,
                       ThreadsafeProducerConsumer<OverlapsAndCigars>& overlaps_and_cigars_to_process,
                       std::atomic<int32_t>& number_of_skipped_pairs_of_indices,
                       cudaStream_t cuda_stream)
{
    GW_NVTX_RANGE(profiler, "main::process_one_batch");
    const IndexBatch& host_batch                  = batch.host_batch;
    const std::vector<IndexBatch>& device_batches = batch.device_batches;

    // if there is only one device batch and it is the same as host bach (which should be the case then) there is no need to copy indices to host
    // as they will be queried only once
    const bool skip_copy_to_host = 1 == device_batches.size();
    assert(!skip_copy_to_host || (host_batch.query_indices == device_batches.front().query_indices && host_batch.target_indices == device_batches.front().target_indices));

    // load indices into host memory
    {
        assert(!host_batch.query_indices.empty() && !host_batch.target_indices.empty() && !device_batches.empty());

        GW_NVTX_RANGE(profiler, "main::process_one_batch::host_indices");
        host_cache.generate_query_cache_content(host_batch.query_indices,
                                                device_batches.front().query_indices,
                                                skip_copy_to_host);
        host_cache.generate_target_cache_content(host_batch.target_indices,
                                                 device_batches.front().target_indices,
                                                 skip_copy_to_host);
    }

    // process device batches one by one
    for (const IndexBatch& device_batch : batch.device_batches)
    {
        process_one_device_batch(device_batch,
                                 device_cache,
                                 application_parameters,
                                 device_allocator,
                                 overlaps_and_cigars_to_process,
                                 number_of_skipped_pairs_of_indices,
                                 cuda_stream);
    }
}

/// \brief does post-processing and writes data to output
/// \param device_id
/// \param application_parameters
/// \param overlaps_and_cigars_to_process new data is added to this structure as it gets available, also signals when there is not going to be any new data
/// \param output_mutex controls access to output to prevent race conditions
void postprocess_and_write_thread_function(const int32_t device_id,
                                           const ApplicationParameters& application_parameters,
                                           ThreadsafeProducerConsumer<OverlapsAndCigars>& overlaps_and_cigars_to_process,
                                           std::mutex& output_mutex)
{
    GW_NVTX_RANGE(profiler, ("main::postprocess_and_write_thread_for_device_" + std::to_string(device_id)).c_str());
    // This function is expected to run in a separate thread so set current device in order to avoid problems
    GW_CU_CHECK_ERR(cudaSetDevice(device_id));

    // keep processing data as it arrives
    gw_optional_t<OverlapsAndCigars> data_to_write;
    while (data_to_write = overlaps_and_cigars_to_process.get_next_element()) // if optional is empty that means that there will be no more overlaps to process and the thread can finish
    {
        {
            GW_NVTX_RANGE(profiler, "main::postprocess_and_write_thread::one_set");
            std::vector<Overlap>& overlaps         = data_to_write->overlaps;
            const std::vector<std::string>& cigars = data_to_write->cigars;

            {
                GW_NVTX_RANGE(profiler, "main::postprocess_and_write_thread::postprocessing");
                // Overlap post processing - add overlaps which can be combined into longer ones.
                Overlapper::post_process_overlaps(data_to_write->overlaps, application_parameters.drop_fused_overlaps);
            }

            if (application_parameters.perform_overlap_end_rescue)
            {
                GW_NVTX_RANGE(profiler, "main::postprocess_and_write_thread::rescue_overlap_end");
                // Perform overlap-end rescue
                Overlapper::rescue_overlap_ends(data_to_write->overlaps,
                                                *application_parameters.query_parser,
                                                *application_parameters.target_parser,
                                                50,
                                                0.5);
            }

            if (application_parameters.all_to_all && application_parameters.drop_self_mappings)
            {
                GW_NVTX_RANGE(profiler, "main::postprocess_and_write_thread::remove_self_mappings");
                ::claraparabricks::genomeworks::cudamapper::details::overlapper::filter_self_mappings(overlaps,
                                                                                                      *application_parameters.query_parser,
                                                                                                      *application_parameters.target_parser,
                                                                                                      0.8);
            }

            // write to output
            {
                GW_NVTX_RANGE(profiler, "main::postprocess_and_write_thread::print_paf");
                print_paf(overlaps,
                          cigars,
                          *application_parameters.query_parser,
                          *application_parameters.target_parser,
                          application_parameters.kmer_size,
                          output_mutex);
            }
        }
    }
}

/// \brief controls one GPU
///
/// Each thread is resposible for one GPU. It takes one batch, processes it and passes it to postprocess_and_write_thread.
/// It keeps doing this as long as there are available batches. It also controls the postprocess_and_write_thread.
///
/// \param device_id
/// \param batches_of_indices
/// \param application_parameters
/// \param output_mutex
/// \param cuda_stream
/// \param number_of_total_batches
/// \param number_of_skipped_pairs_of_indices
/// \param number_of_processed_batches
void worker_thread_function(const int32_t device_id,
                            ThreadsafeDataProvider<BatchOfIndices>& batches_of_indices,
                            const ApplicationParameters& application_parameters,
                            std::mutex& output_mutex,
                            cudaStream_t cuda_stream,
                            const int64_t number_of_total_batches,
                            std::atomic<int32_t>& number_of_skipped_pairs_of_indices,
                            std::atomic<int64_t>& number_of_processed_batches)
{
    GW_NVTX_RANGE(profiler, "main::worker_thread");

    // This function is expected to run in a separate thread so set current device in order to avoid problems
    GW_CU_CHECK_ERR(cudaSetDevice(device_id));

    DefaultDeviceAllocator device_allocator = create_default_device_allocator(application_parameters.max_cached_memory_bytes);

    // create host_cache, data is not loaded at this point but later as each batch gets processed
    auto host_cache = std::make_shared<IndexCacheHost>(application_parameters.all_to_all,
                                                       device_allocator,
                                                       application_parameters.query_parser,
                                                       application_parameters.target_parser,
                                                       application_parameters.kmer_size,
                                                       application_parameters.windows_size,
                                                       true, // hash_representations
                                                       application_parameters.filtering_parameter,
                                                       cuda_stream);

    // create host_cache, data is not loaded at this point but later as each batch gets processed
    IndexCacheDevice device_cache(application_parameters.all_to_all,
                                  host_cache);

    // data structure used to exchange data with postprocess_and_write_thread
    ThreadsafeProducerConsumer<OverlapsAndCigars> overlaps_and_cigars_to_process;

    // There should be at least one postprocess_and_write_thread per worker_thread. If more threads are available one thread should be reserved for
    // worker_thread and all other threads should be postprocess_and_write_threads
    const int32_t threads_per_device = ceiling_divide(static_cast<int32_t>(std::thread::hardware_concurrency()),
                                                      application_parameters.num_devices);

    const int32_t postprocess_and_write_threads_per_device = std::max(threads_per_device - 1, 1);

    // postprocess_and_write_threads run in the background and post-process and write overlaps and cigars to output as they become available in overlaps_and_cigars_to_process
    std::vector<std::thread> postprocess_and_write_threads;
    for (int32_t i = 0; i < postprocess_and_write_threads_per_device; ++i)
    {
        postprocess_and_write_threads.emplace_back(postprocess_and_write_thread_function,
                                                   device_id,
                                                   std::ref(application_parameters),
                                                   std::ref(overlaps_and_cigars_to_process),
                                                   std::ref(output_mutex));
    }

    // keep processing batches of indices until there are none left
    gw_optional_t<BatchOfIndices> batch_of_indices;
    while (batch_of_indices = batches_of_indices.get_next_element()) // if optional is empty that means that there are no more batches to process and the thread can finish
    {
        const int64_t batch_number         = number_of_processed_batches.fetch_add(1); // as this is not called atomically with get_next_element() the value does not have to be completely accurate, but this is ok as the value is only use for displaying progress
        const std::string progress_message = "Device " + std::to_string(device_id) + " took batch " + std::to_string(batch_number + 1) + " out of " + std::to_string(number_of_total_batches) + " batches in total\n";
        std::cerr << progress_message; // TODO: possible race condition, switch to logging library

        process_one_batch(batch_of_indices.value(),
                          application_parameters,
                          device_allocator,
                          *host_cache,
                          device_cache,
                          overlaps_and_cigars_to_process,
                          number_of_skipped_pairs_of_indices,
                          cuda_stream);
    }

    // tell writer thread that there will be no more overlaps and it can finish once it has written all overlaps
    overlaps_and_cigars_to_process.signal_pushed_last_element();

    for (std::thread& postprocess_and_write_thread : postprocess_and_write_threads)
    {
        postprocess_and_write_thread.join();
    }

    // by this point all GPU work should anyway be done as postprocess_and_write_thread also finished and all GPU work had to be done before last values could be written
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
}

} // namespace

int main(int argc, char* argv[])
{
    logging::Init();

    const ApplicationParameters parameters(argc, argv);

    std::mutex output_mutex;

    // Program should process all combinations of query and target (if query and target are the same half of those can be skipped
    // due to symmetry). The matrix of query-target combinations is split into tiles called batches. Worker threads (one per GPU)
    // take batches one by one and process them.
    // Every batch is small enough for its indices to fit in host memory. Batches are further divided into sub-batches which are
    // small enough that all their indices fit in device memory.
    // After a worker thread has taken a batch it generates all necessary indices and saves them in host memory using IndexCacheHost.
    // It then processes sub-batches one by one but first loading indices into IndexCacheDevice from IndexCacheHost and then finding
    // the overlaps.
    // Output formatting and writing is done by a separate thread.

    // Split work into batches
    std::vector<BatchOfIndices> batches_of_indices_vect = generate_batches_of_indices(parameters.query_indices_in_host_memory,
                                                                                      parameters.query_indices_in_device_memory,
                                                                                      parameters.target_indices_in_host_memory,
                                                                                      parameters.target_indices_in_device_memory,
                                                                                      parameters.query_parser,
                                                                                      parameters.target_parser,
                                                                                      parameters.index_size * 1'000'000,        // value was in MB
                                                                                      parameters.target_index_size * 1'000'000, // value was in MB
                                                                                      parameters.all_to_all);
    const int64_t number_of_total_batches               = get_size<int64_t>(batches_of_indices_vect);
    std::atomic<int64_t> number_of_processed_batches(0);
    ThreadsafeDataProvider<BatchOfIndices> batches_of_indices(std::move(batches_of_indices_vect));

    // pairs of indices might be skipped if they cause out of memory errors
    std::atomic<int32_t> number_of_skipped_pairs_of_indices{0};

    // create worker threads (one thread per device)
    // these thread process batches_of_indices one by one
    std::vector<std::thread> worker_threads;

    // CudaStreams for each thread
    std::vector<CudaStream> cuda_streams;

    for (int32_t device_id = 0; device_id < parameters.num_devices; ++device_id)
    {
        GW_CU_CHECK_ERR(cudaSetDevice(device_id));
        cuda_streams.emplace_back(make_cuda_stream());
        worker_threads.emplace_back(worker_thread_function,
                                    device_id,
                                    std::ref(batches_of_indices),
                                    std::ref(parameters),
                                    std::ref(output_mutex),
                                    cuda_streams.back().get(),
                                    number_of_total_batches,
                                    std::ref(number_of_skipped_pairs_of_indices),
                                    std::ref(number_of_processed_batches));
    }

    // wait for all work to be done
    for (auto& t : worker_threads)
    {
        // no need to sync, it should be done at the end of worker_threads
        t.join();
    }

    if (number_of_skipped_pairs_of_indices != 0)
    {
        std::cerr << "NOTE: Skipped " << number_of_skipped_pairs_of_indices << " pairs of indices due to device out of memory error" << std::endl;
    }

    return 0;
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks

/// \brief main function
/// main function cannot be in a namespace so using this function to call actual main function
int main(int argc, char* argv[])
{
    return claraparabricks::genomeworks::cudamapper::main(argc, argv);
}
