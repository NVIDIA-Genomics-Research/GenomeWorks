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
* Sneha Goenka (gsneha@stanford.edu) and Yatish Turakhia (yturakhi@uscs.edu).
* Source code for original implementation and use in SegAlign can be found
* here: https://github.com/gsneha26/SegAlign
*/
#pragma once

#include <claraparabricks/genomeworks/cudaextender/extender.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>
#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

class UngappedXDrop : public Extender
{
public:
    UngappedXDrop(int32_t* h_score_mat,
                  int32_t score_mat_dim,
                  int32_t xdrop_threshold,
                  bool no_entropy,
                  cudaStream_t stream,
                  int32_t device_id,
                  DefaultDeviceAllocator allocator);
    ~UngappedXDrop() override;

    StatusType extend_async(int8_t* h_query, int32_t query_length,
                            int8_t* h_target, int32_t target_length,
                            int32_t score_threshold,
                            const std::vector<SeedPair>& h_seed_pairs) override;

    StatusType extend_async(int8_t* d_query, int32_t query_length,
                            int8_t* d_target, int32_t target_length,
                            int32_t score_threshold, SeedPair* d_seed_pairs,
                            int32_t num_seed_pairs, ScoredSegmentPair* d_scored_segment_pairs,
                            int32_t* d_num_scored_segment_pairs) override;

    StatusType sync() override;

    void reset() override;

    const std::vector<ScoredSegmentPair>& get_scored_segment_pairs() const override;

private:
    DefaultDeviceAllocator allocator_;
    // Device ptr API required variables
    pinned_host_vector<int32_t> h_score_mat_;
    const int32_t score_mat_dim_; // Assume matrix is square
    const int32_t xdrop_threshold_;
    const bool no_entropy_;
    cudaStream_t stream_;
    const int32_t device_id_;
    std::vector<ScoredSegmentPair> scored_segment_pairs_;
    int32_t batch_max_ungapped_extensions_;      // TODO - Make const
    device_buffer<int32_t> d_score_mat_;         // Pointer to device substitution matrix
    device_buffer<int32_t> d_done_;              // TODO- Rename scratch space
    device_buffer<ScoredSegmentPair> d_tmp_ssp_; // TODO- Rename Scratch space 2
    int32_t total_scored_segment_pairs_;
    device_buffer<int8_t> d_temp_storage_cub_; // temporary storage for cub functions

    // Host ptr API additional required variables
    bool host_ptr_api_mode_;
    device_buffer<int8_t> d_query_;
    device_buffer<int8_t> d_target_;
    device_buffer<SeedPair> d_seed_pairs_;
    device_buffer<int32_t> d_num_ssp_;
    device_buffer<ScoredSegmentPair> d_ssp_;
    std::vector<ScoredSegmentPair> h_ssp_;
};

} // namespace cudaextender

} // namespace genomeworks

} // namespace claraparabricks
