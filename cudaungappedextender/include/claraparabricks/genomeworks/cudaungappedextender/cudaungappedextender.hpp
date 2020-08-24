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
#pragma once

namespace claraparabricks
{
namespace genomeworks
{
namespace cudaungappedextender
{

struct Anchor
{
    /// position of first sketch element in query_read_id_
    position_in_read_t query_position_in_read_;
    /// position of second sketch element in target_read_id_
    position_in_read_t target_position_in_read_;
};

struct ScoredSegment
{
    /// Anchor for the segment
    Anchor anchor;
    /// length of the segment
    int32_t len;
    /// score of the segment
    int32_t score;
};

/// CUDA Ungapped Extension object
class UngappedExtender
{
public:
    /// \brief Destructor for UngappedExtender.
    ~UngappedExtender() = default;
    /// \brief Constructor Prototype
    UngappedExtender(int* h_sub_mat, int input_xdrop, bool input_noentropy,
                     int gpu_id = 0, cudaStream_t stream = 0);

    /// \brief Host pointer prototype for ungapped extension
    ///
    /// Takes values from host data structures,
    /// copies them over to device,
    /// launches async extension kernels on specified stream. Filters
    /// segments on device based on input_hspthresh.
    StatusType ungapped_extend(const char* h_query, int32_t query_length,
                               const char* h_target, int32_t target_length,
                               int32_t input_hspthresh,
                               std::vector<Anchor>& h_seedHits);

    /// \brief Device pointer prototype for ungapped extension
    ///
    /// Memcopies to device memory are assumed to be done before this
    /// function. Output array d_hsp_out must be pre-allocated on device.
    /// Launches async extension kernel. Filters segments on device
    /// based on input_hspthresh.
    StatusType ungapped_extend(const char* d_query, int32_t query_length,
                               const char* d_target, int32_t target_length,
                               int32_t input_hspthresh, Anchor* d_seed_hits,
                               int32_t num_hits, ScoredSegment* d_hsp_out,
                               int32_t* d_num_hsps);

    /// \brief Waits for CUDA accelerated extension to finish
    ///
    /// Blocking call that waits for all the extensions scheduled
    /// on the GPU to come to completion.
    StatusType sync_extensions();

    /// \brief Return the computed segments
    ///
    /// \return Vector of Scored Segments
    const std::vector<std::shared_ptr<ScoredSegment>>& get_scored_segments();

    /// \brief Reset UngappedExtender object.
    void reset();
};
} // namespace cudaungappedextender
} // namespace genomeworks
} // namespace claraparabricks
