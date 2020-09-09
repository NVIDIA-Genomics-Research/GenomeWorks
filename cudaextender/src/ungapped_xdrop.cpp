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
#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include "ungapped_xdrop.hpp"
#include "ungapped_xdrop_kernels.cuh"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cpp/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

using namespace cudautils;

UngappedXDrop::UngappedXDrop(int32_t* h_sub_mat, int32_t sub_mat_dim, int32_t xdrop_threshold, bool no_entropy, cudaStream_t stream, int32_t device_id)
    : h_sub_mat_(h_sub_mat)
    , sub_mat_dim_(sub_mat_dim)
    , xdrop_threshold_(xdrop_threshold)
    , no_entropy_(no_entropy)
    , stream_(stream)
    , device_id_(device_id)
{
    // Calculate the max limits on the number of extensions we can do on
    // this GPU
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id_);
    const float global_mem_gb         = static_cast<float>(device_prop.totalGlobalMem / 1073741824.0f);
    const int32_t max_ungapped_per_gb = 4194304; // FIXME: Calculate using sizeof datastructures
    max_ungapped_extensions_          = static_cast<int32_t>(global_mem_gb * max_ungapped_per_gb);

    // Switch to device for copying over initial structures
    scoped_device_switch dev(device_id_);

    // Allocate space on device for scoring matrix and
    GW_CU_CHECK_ERR(cudaMalloc((void**)&d_sub_mat_, sub_mat_dim_ * sizeof(int32_t)));
    GW_CU_CHECK_ERR(cudaMalloc((void**)&d_done_, max_ungapped_extensions_ * sizeof(int32_t)));
    GW_CU_CHECK_ERR(cudaMalloc((void**)&d_tmp_hsp_, max_ungapped_extensions_ * sizeof(ScoredSegmentPair)));

    // FIXME - Pinned host memory registration for proper async behavior
    device_copy_n(h_sub_mat_, sub_mat_dim_ * sizeof(int32_t), d_sub_mat_, stream_);
    GW_CU_CHECK_ERR(cudaMemsetAsync((void*)d_done_, 0, max_ungapped_extensions_ * sizeof(int32_t), stream_));
    GW_CU_CHECK_ERR(cudaMemsetAsync((void*)d_tmp_hsp_, 0, max_ungapped_extensions_ * sizeof(ScoredSegmentPair), stream_));
}

StatusType UngappedXDrop::extend_async(const char* d_query, int32_t query_length, const char* d_target, int32_t target_length, int32_t score_threshold, SeedPair* d_seed_pairs, int32_t num_seed_pairs, ScoredSegmentPair* d_scored_segment_pairs, int32_t* d_num_scored_segment_pairs)
{
    // Switch to configured GPU
    scoped_device_switch dev(device_id_);

    int32_t curr_num_pairs = 0;
    int32_t num_anchors = 0;
    for (int32_t seed_pair_start = 0; seed_pair_start < num_seed_pairs; seed_pair_start += max_ungapped_extensions_)
    {
        curr_num_pairs = std::min(max_ungapped_extensions_, num_seed_pairs - seed_pair_start);
        // TODO- Extricate the kernel params?
        find_high_scoring_segment_pairs<<<1024, 128, stream_>>>(d_query, query_length, d_target, target_length, d_sub_mat_, no_entropy_, xdrop_threshold_, score_threshold, curr_num_pairs, d_seed_pairs, seed_pair_start, d_scored_segment_pairs, d_done);
        thrust::device_ptr<int32_t> d_done_dev_ptr = thrust::device_pointer_cast(d_done_scratch_);
        // TODO- Make thrust use caching allocator or change kernel
        thrust::inclusive_scan(thrust::cuda::par.on(stream_), d_done_dev_ptr, d_done_dev_ptr + curr_num_pairs, d_done_dev_ptr);
        device_copy_n((void*)(d_done[gpu_id]+curr_num_hits-1), sizeof(int32_t), &num_anchors, stream_);
        // TODO- Make async
        cudaStreamSynchronize(stream_);
        if(num_anchors > 0){
            compress_output<<<1024, 1024, stream_>>>(d_done_, seed_pair_start, scored_segment_pairs_, d_tmp_hsp_, curr_num_pairs); // TODO- Need configurability for kernel?

            thrust::stable_sort(thrust::cuda::par.on(stream_), d_tmp_hsp_vec[gpu_id].begin(), d_tmp_hsp_vec[gpu_id].begin()+num_anchors, hspComp());
            thrust::device_vector<segment>::iterator result_end = thrust::unique_copy(thrust::cuda::par.on(stream_), d_tmp_hsp_vec[gpu_id].begin(), d_tmp_hsp_vec[gpu_id].begin()+num_anchors, d_hsp_vec[gpu_id].begin()+total_anchors,  hspEqual());
            num_anchors = thrust::distance(thrust::cuda::par.on(stream_), d_hsp_vec[gpu_id].begin()+total_anchors, result_end), num_anchors;
            total_anchors += num_anchors;
        }
    }
    return success;
}

} // namespace cudaextender

} // namespace genomeworks

} // namespace claraparabricks