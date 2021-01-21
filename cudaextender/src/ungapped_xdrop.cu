/*
* Copyright 2020 NVIDIA CORPORATION.
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
/*
* This algorithm was adapted from SegAlign's Ungapped Extender authored by
* Sneha Goenka (gsneha@stanford.edu) and Yatish Turakhia (yturakhi@ucsc.edu).
* Source code for original implementation and use in SegAlign can be found
* here: https://github.com/gsneha26/SegAlign
* Description of the algorithm and original implementation can be found in the SegAlign 
* paper published in SC20 (https://doi.ieeecomputersociety.org/10.1109/SC41405.2020.00043)
*/
#include "ungapped_xdrop.cuh"
#include "ungapped_xdrop_kernels.cuh"

#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

using namespace cudautils;

UngappedXDrop::UngappedXDrop(const int32_t* h_score_mat, const int32_t score_mat_dim,
                             const int32_t xdrop_threshold, const bool no_entropy,
                             cudaStream_t stream, const int32_t device_id,
                             DefaultDeviceAllocator allocator)
    : h_score_mat_(h_score_mat, h_score_mat + score_mat_dim)
    , score_mat_dim_(score_mat_dim)
    , xdrop_threshold_(xdrop_threshold)
    , no_entropy_(no_entropy)
    , stream_(stream)
    , device_id_(device_id)
    , host_ptr_api_mode_(false)
    , allocator_(allocator)
{
    // Switch to device for copying over initial structures
    scoped_device_switch dev(device_id_);

    // Calculate the max limits on the number of extensions we can do on this GPU
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id_);

    // TODO - Currently element and memory limits are artifacts of hardcoded global memory limits in
    // SegAlign. To be replaced with actual calculation of memory requirements with sizes of
    // datastructures taken into consideration. Also currently the max limit is based on total
    // global memory, which should be replaced with memory available from the passed in allocator.
    // Github Issue: https://github.com/clara-parabricks/GenomeWorks/issues/576
    const int32_t max_ungapped_per_gb = 4194304;
    const float global_mem_gb         = static_cast<float>(device_prop.totalGlobalMem) / 1073741824.0f;
    batch_max_ungapped_extensions_    = static_cast<int32_t>(global_mem_gb) * max_ungapped_per_gb;

    //Figure out memory requirements for cub functions
    size_t temp_storage_bytes = 0;
    size_t cub_storage_bytes  = 0;
    GW_CU_CHECK_ERR(cub::DeviceSelect::Unique(nullptr,
                                              temp_storage_bytes,
                                              d_tmp_ssp_.data(),
                                              d_tmp_ssp_.data(),
                                              (int32_t*)nullptr,
                                              batch_max_ungapped_extensions_,
                                              stream_));
    GW_CU_CHECK_ERR(cub::DeviceScan::InclusiveSum(nullptr,
                                                  cub_storage_bytes,
                                                  d_done_.data(),
                                                  d_done_.data(),
                                                  batch_max_ungapped_extensions_,
                                                  stream_));
    cub_storage_bytes = std::max(temp_storage_bytes, cub_storage_bytes);

    // Allocate space on device for scoring matrix and intermediate results
    d_score_mat_        = device_buffer<int32_t>(score_mat_dim_, allocator_, stream_);
    d_done_             = device_buffer<int32_t>(batch_max_ungapped_extensions_, allocator_, stream_);
    d_tmp_ssp_          = device_buffer<ScoredSegmentPair>(batch_max_ungapped_extensions_, allocator_, stream_);
    d_temp_storage_cub_ = device_buffer<int8_t>(cub_storage_bytes, allocator_, stream_);
    device_copy_n(h_score_mat_.data(), score_mat_dim_, d_score_mat_.data(), stream_);
}

StatusType UngappedXDrop::extend_async(const int8_t* d_query, const int32_t query_length,
                                       const int8_t* d_target, const int32_t target_length,
                                       const int32_t score_threshold, const SeedPair* d_seed_pairs,
                                       const int32_t num_seed_pairs, ScoredSegmentPair* d_scored_segment_pairs,
                                       int32_t* d_num_scored_segment_pairs)
{
    if (d_query == nullptr || d_target == nullptr || d_seed_pairs == nullptr)
    {
        GW_LOG_ERROR("Invalid input pointers");
        return StatusType::invalid_input;
    }
    if (d_scored_segment_pairs == nullptr || d_num_scored_segment_pairs == nullptr)
    {
        GW_LOG_ERROR("Invalid output pointers");
        return StatusType::invalid_input;
    }
    // Switch to configured GPU
    scoped_device_switch dev(device_id_);
    total_scored_segment_pairs_ = 0;
    for (int32_t seed_pair_start = 0; seed_pair_start < num_seed_pairs; seed_pair_start += batch_max_ungapped_extensions_)
    {
        // TODO - Kernel optimizations [Unnecessary memset?]
        // Github Issue: https://github.com/clara-parabricks/GenomeWorks/issues/579
        GW_CU_CHECK_ERR(cudaMemsetAsync((void*)d_done_.data(), 0, batch_max_ungapped_extensions_ * sizeof(int32_t), stream_));
        GW_CU_CHECK_ERR(cudaMemsetAsync((void*)d_tmp_ssp_.data(), 0, batch_max_ungapped_extensions_ * sizeof(ScoredSegmentPair), stream_));
        const int32_t curr_num_pairs = std::min(batch_max_ungapped_extensions_, num_seed_pairs - seed_pair_start);
        find_high_scoring_segment_pairs<<<1024, 128, 0, stream_>>>(d_target,
                                                                   target_length,
                                                                   d_query,
                                                                   query_length,
                                                                   d_score_mat_.data(),
                                                                   no_entropy_,
                                                                   xdrop_threshold_,
                                                                   score_threshold,
                                                                   d_seed_pairs,
                                                                   curr_num_pairs,
                                                                   seed_pair_start,
                                                                   d_scored_segment_pairs,
                                                                   d_done_.data());
        size_t cub_storage_bytes = d_temp_storage_cub_.size();
        GW_CU_CHECK_ERR(cub::DeviceScan::InclusiveSum(d_temp_storage_cub_.data(),
                                                      cub_storage_bytes,
                                                      d_done_.data(),
                                                      d_done_.data(),
                                                      curr_num_pairs,
                                                      stream_))
        // TODO- Make output compression async. Currently synchronocity is arising due to
        // thrust::stable_sort. Dynamic parallelism or an equivalent sort with cub can be used
        // Github Issue: https://github.com/clara-parabricks/GenomeWorks/issues/578
        const int32_t num_scored_segment_pairs = get_value_from_device(d_done_.data() + curr_num_pairs - 1, stream_);
        if (num_scored_segment_pairs > 0)
        {
            // TODO - Explore scaling up/down launch config based on workload. Also explore making
            // this accessible to the user for configuration
            // Github Issue: https://github.com/clara-parabricks/GenomeWorks/issues/577
            compress_output<<<1024, 1024, 0, stream_>>>(d_done_.data(),
                                                        seed_pair_start,
                                                        d_scored_segment_pairs,
                                                        d_tmp_ssp_.data(),
                                                        curr_num_pairs);
            thrust::stable_sort(thrust::cuda::par(allocator_).on(stream_),
                                d_tmp_ssp_.begin(),
                                d_tmp_ssp_.begin() + num_scored_segment_pairs,
                                scored_segment_pair_comp());

            ScoredSegmentPair* result_end =
                thrust::unique_copy(thrust::cuda::par(allocator_).on(stream_),
                                    d_tmp_ssp_.begin(),
                                    d_tmp_ssp_.begin() + num_scored_segment_pairs,
                                    d_scored_segment_pairs + total_scored_segment_pairs_,
                                    scored_segment_pair_equal());

            total_scored_segment_pairs_ += thrust::distance(
                d_scored_segment_pairs + total_scored_segment_pairs_,
                result_end);
        }
    }

    set_device_value_async(d_num_scored_segment_pairs, &total_scored_segment_pairs_, stream_);

    return StatusType::success;
}

StatusType UngappedXDrop::extend_async(const int8_t* h_query, const int32_t query_length,
                                       const int8_t* h_target, const int32_t target_length,
                                       const int32_t score_threshold,
                                       const std::vector<SeedPair>& h_seed_pairs)
{
    // Reset the extender if it was used before in this mode
    reset();
    // Set host pointer mode on
    host_ptr_api_mode_ = true;
    // Allocate space for query and target sequences
    d_query_  = device_buffer<int8_t>(query_length, allocator_, stream_);
    d_target_ = device_buffer<int8_t>(target_length, allocator_, stream_);
    // Allocate space for SeedPair input
    d_seed_pairs_ = device_buffer<SeedPair>(h_seed_pairs.size(), allocator_, stream_);
    // Allocate space for ScoredSegmentPair output
    d_ssp_     = device_buffer<ScoredSegmentPair>(h_seed_pairs.size(), allocator_, stream_);
    d_num_ssp_ = device_buffer<int32_t>(1, allocator_, stream_);

    // Async memcopy all the input values to device
    device_copy_n(h_query, query_length, d_query_.data(), stream_);
    device_copy_n(h_target, target_length, d_target_.data(), stream_);
    device_copy_n(h_seed_pairs.data(), h_seed_pairs.size(), d_seed_pairs_.data(), stream_);

    // Launch the ungapped extender device function
    return extend_async(d_query_.data(), query_length,
                        d_target_.data(), target_length,
                        score_threshold, d_seed_pairs_.data(),
                        d_seed_pairs_.size(), d_ssp_.data(),
                        d_num_ssp_.data());
}

StatusType UngappedXDrop::sync()
{
    if (host_ptr_api_mode_)
    {
        const int32_t h_num_ssp = get_value_from_device(d_num_ssp_.data(), stream_);
        if (h_num_ssp > 0)
        {
            h_ssp_.resize(h_num_ssp);
            device_copy_n(d_ssp_.data(), h_num_ssp, h_ssp_.data(), stream_);
            GW_CU_CHECK_ERR(cudaStreamSynchronize(stream_));
        }
        return StatusType::success;
    }

    // If this function was called without using the host pointer API, throw error
    return StatusType::invalid_operation;
}

const std::vector<ScoredSegmentPair>& UngappedXDrop::get_scored_segment_pairs() const
{
    if (host_ptr_api_mode_)
    {
        return h_ssp_;
    }
    // If this function was called using the host pointer API, throw error
    throw std::runtime_error("Invalid API call. Getting scored segment pairs without calling extend_async host ptr API");
}

void UngappedXDrop::reset()
{
    // Reset these only if host pointer API was used earlier
    if (host_ptr_api_mode_)
    {
        h_ssp_.clear();
        host_ptr_api_mode_ = false;
    }
}

UngappedXDrop::~UngappedXDrop()
{
    UngappedXDrop::reset();
}

} // namespace cudaextender

} // namespace genomeworks

} // namespace claraparabricks
