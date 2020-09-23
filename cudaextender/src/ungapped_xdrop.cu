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
#include "ungapped_xdrop.cuh"
#include "ungapped_xdrop_kernels.cuh"

#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>

#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

using namespace cudautils;

UngappedXDrop::UngappedXDrop(const int32_t* h_sub_mat, const int32_t sub_mat_dim, const int32_t xdrop_threshold, const bool no_entropy, cudaStream_t stream, const int32_t device_id, DefaultDeviceAllocator allocator)
    : h_sub_mat_(h_sub_mat)
    , sub_mat_dim_(sub_mat_dim)
    , xdrop_threshold_(xdrop_threshold)
    , no_entropy_(no_entropy)
    , stream_(stream)
    , device_id_(device_id)
    , host_ptr_api_mode_(false)
    , allocator_(allocator)
{
    if (h_sub_mat_ == nullptr)
    {
        throw std::runtime_error("Substitution matrix cannot be null");
    }
    // TODO - check sub_mat_dim based on Sequence Encoder API
    // Calculate the max limits on the number of extensions we can do on this GPU
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id_);
    const int32_t max_ungapped_per_gb = 4194304; // FIXME: Calculate using sizeof datastructures
    //const int32_t max_seed_pairs_per_gb = 8388608; // FIXME: Calculate using sizeof datastructures // TODO- Do we need this?
    const float global_mem_gb      = static_cast<float>(device_prop.totalGlobalMem) / 1073741824.0f;
    batch_max_ungapped_extensions_ = static_cast<int32_t>(global_mem_gb) * max_ungapped_per_gb;
    // Switch to device for copying over initial structures
    scoped_device_switch dev(device_id_);

    //Figure out memory requirements for cub functions
    size_t temp_storage_bytes = 0;
    size_t cub_storage_bytes  = 0;
    GW_CU_CHECK_ERR(cub::DeviceSelect::Unique(nullptr, temp_storage_bytes, d_tmp_ssp_.data(), d_tmp_ssp_.data(), (int32_t*)nullptr, batch_max_ungapped_extensions_, stream_));
    GW_CU_CHECK_ERR(cub::DeviceScan::InclusiveSum(nullptr, cub_storage_bytes, d_done_.data(), d_done_.data(), batch_max_ungapped_extensions_, stream_));
    cub_storage_bytes = std::max(temp_storage_bytes, cub_storage_bytes);

    // Allocate space on device for scoring matrix and intermediate results
    d_sub_mat_          = device_buffer<int32_t>(sub_mat_dim_, allocator_, stream_);
    d_done_             = device_buffer<int32_t>(batch_max_ungapped_extensions_, allocator_, stream_);
    d_tmp_ssp_          = device_buffer<ScoredSegmentPair>(batch_max_ungapped_extensions_, allocator_, stream_);
    d_temp_storage_cub_ = device_buffer<char>(cub_storage_bytes, allocator_, stream_);

    // Requires pinned host memory registration for proper async behavior
    device_copy_n(h_sub_mat_, sub_mat_dim_, d_sub_mat_.data(), stream_);
}

StatusType UngappedXDrop::extend_async(const char* d_query, int32_t query_length,
                                       const char* d_target, int32_t target_length,
                                       int32_t score_threshold, SeedPair* d_seed_pairs,
                                       int32_t num_seed_pairs, ScoredSegmentPair* d_scored_segment_pairs,
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
        // TODO - Do we need these? It seems we don't!
        GW_CU_CHECK_ERR(cudaMemsetAsync((void*)d_done_.data(), 0, batch_max_ungapped_extensions_ * sizeof(int32_t), stream_));
        GW_CU_CHECK_ERR(cudaMemsetAsync((void*)d_tmp_ssp_.data(), 0, batch_max_ungapped_extensions_ * sizeof(ScoredSegmentPair), stream_));
        const int32_t curr_num_pairs = std::min(batch_max_ungapped_extensions_, num_seed_pairs - seed_pair_start);
        // TODO- Extricate the kernel launch params?
        find_high_scoring_segment_pairs<<<1024, 128, 0, stream_>>>(d_target,
                                                                   target_length,
                                                                   d_query,
                                                                   query_length,
                                                                   d_sub_mat_.data(),
                                                                   no_entropy_,
                                                                   xdrop_threshold_,
                                                                   score_threshold,
                                                                   d_seed_pairs,
                                                                   curr_num_pairs,
                                                                   seed_pair_start,
                                                                   d_scored_segment_pairs,
                                                                   d_done_.data());
        size_t cub_storage_bytes = d_temp_storage_cub_.size();
        GW_CU_CHECK_ERR(cub::DeviceScan::InclusiveSum(d_temp_storage_cub_.data(), cub_storage_bytes, d_done_.data(), d_done_.data(), curr_num_pairs, stream_))
        // TODO- Make async
        const int32_t num_scored_segment_pairs = get_value_from_device(d_done_.data() + curr_num_pairs - 1, stream_);
        if (num_scored_segment_pairs > 0)
        {
            compress_output<<<1024, 1024, 0, stream_>>>(d_done_.data(),
                                                        seed_pair_start,
                                                        d_scored_segment_pairs,
                                                        d_tmp_ssp_.data(),
                                                        curr_num_pairs); // TODO- Need configurability for kernel?
            thrust::stable_sort(thrust::cuda::par(allocator_).on(stream_),
                                d_tmp_ssp_.begin(),
                                d_tmp_ssp_.begin() + num_scored_segment_pairs,
                                scored_segment_pair_comp());
            GW_CU_CHECK_ERR(cub::DeviceSelect::Unique(d_temp_storage_cub_.data(),
                                                      cub_storage_bytes,
                                                      d_tmp_ssp_.data(),
                                                      d_scored_segment_pairs + total_scored_segment_pairs_,
                                                      d_num_scored_segment_pairs,
                                                      num_scored_segment_pairs,
                                                      stream_))
            total_scored_segment_pairs_ += get_value_from_device(d_num_scored_segment_pairs, stream_);
        }
    }

    set_device_value_async(d_num_scored_segment_pairs, &total_scored_segment_pairs_, stream_);

    return StatusType::success;
}

StatusType UngappedXDrop::extend_async(const char* h_query, const int32_t& query_length,
                                       const char* h_target, const int32_t& target_length,
                                       const int32_t& score_threshold,
                                       const std::vector<SeedPair>& h_seed_pairs)
{
    // Reset the extender if it was used before in this mode
    reset();
    // Set host pointer mode on
    host_ptr_api_mode_ = true;
    // Allocate space for query and target sequences
    d_query_  = device_buffer<char>(query_length, allocator_, stream_);
    d_target_ = device_buffer<char>(target_length, allocator_, stream_);
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
            cudaStreamSynchronize(stream_);
        }
        return StatusType::success;
    }

    // If this function was called without using the host_ptr_api, throw error
    return StatusType::invalid_operation;
}

const std::vector<ScoredSegmentPair>& UngappedXDrop::get_scored_segment_pairs() const
{
    if (host_ptr_api_mode_)
    {
        return h_ssp_;
    }
    // If this function was called using the host_ptr_api, throw error
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
