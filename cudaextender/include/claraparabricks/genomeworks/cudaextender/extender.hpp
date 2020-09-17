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
#include <claraparabricks/genomeworks/types.hpp>
#include <claraparabricks/genomeworks/cudaextender/cudaextender.hpp>

#include <vector>
#include <memory>
#include <cuda_runtime_api.h>

namespace claraparabricks
{
namespace genomeworks
{
namespace cudaextender
{

typedef struct SeedPair
{
    /// position of first sketch element in query_read_id_
    position_in_read_t query_position_in_read;
    /// position of second sketch element in target_read_id_
    position_in_read_t target_position_in_read;
} SeedPair;

typedef struct ScoredSegmentPair
{
    /// Seed for the segment
    SeedPair seed_pair;
    /// length of the segment
    int32_t length;
    /// score of the segment
    int32_t score;

    __host__ __device__ bool operator==(const ScoredSegmentPair& other) const
    {
        return ((seed_pair.target_position_in_read == other.seed_pair.target_position_in_read) && (seed_pair.query_position_in_read == other.seed_pair.query_position_in_read) && (length == other.length) && (score == other.score));
    }
} ScoredSegmentPair;

/// CUDA Extension object
class Extender
{
public:
    /// \brief Virtual destructor for Extender.
    virtual ~Extender() = default;

    /// \brief Host pointer prototype for extension
    ///
    /// Takes values from host data structures,
    /// copies them over to device,
    /// launches async extension kernels on specified stream. Filters
    /// segments on device based on score_threshold
    virtual StatusType extend_async(const char* h_query, int32_t query_length,
                                    const char* h_target, int32_t target_length,
                                    int32_t score_threshold,
                                    std::vector<SeedPair>& h_seed_pairs) = 0;

    /// \brief Device pointer prototype for  extension
    ///
    /// Memcopies to device memory are assumed to be done before this
    /// function. Output array d_hsp_out must be pre-allocated on device.
    /// Launches async extension kernel. Filters segments on device
    /// based on input_hspthresh.
    virtual StatusType extend_async(const char* d_query, int32_t query_length,
                                    const char* d_target, int32_t target_length,
                                    int32_t score_threshold, SeedPair* d_seed_pairs,
                                    int32_t num_seed_pairs, ScoredSegmentPair* d_scored_segment_pairs,
                                    int32_t& num_scored_segment_pairs) = 0;

    /// \brief Waits for CUDA accelerated extension to finish
    ///
    /// Blocking call that waits for all the extensions scheduled
    /// on the GPU to come to completion.
    virtual StatusType sync() = 0;

    /// \brief Return the computed segments
    ///
    /// \return Vector of Scored Segments
    virtual const std::vector<ScoredSegmentPair>& get_scored_segment_pairs() const = 0;

    /// \brief Reset Extender object and free device/host memory
    virtual void reset() = 0;
};

std::unique_ptr<Extender> create_extender(int32_t* h_sub_mat, int32_t sub_mat_dim, int32_t xdrop_threshold, bool no_entropy, cudaStream_t stream, int32_t device_id, ExtensionType type=ExtensionType::ungapped_xdrop);

} // namespace cudaextender
} // namespace genomeworks
} // namespace claraparabricks
