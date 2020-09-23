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
#include <claraparabricks/genomeworks/cudaextender/cudaextender.hpp>
#include <claraparabricks/genomeworks/types.hpp>
#include <claraparabricks/genomeworks/utils/allocator.hpp>
#include <cuda_runtime_api.h>
#include <vector>
#include <memory>

namespace claraparabricks
{
namespace genomeworks
{
namespace cudaextender
{

/// Seed positions in target and query reads
struct SeedPair
{
    /// position of first sketch element in query_read_id_
    position_in_read_t query_position_in_read;
    /// position of second sketch element in target_read_id_
    position_in_read_t target_position_in_read;
};

/// Segment pairs in target & query reads with associated length & score
struct ScoredSegmentPair
{
    /// seed for the segment
    SeedPair seed_pair;
    /// length of the segment
    int32_t length;
    /// score of the segment
    int32_t score;
    /// equality operator
    __host__ __device__ bool operator==(const ScoredSegmentPair& other) const
    {
        return ((seed_pair.target_position_in_read == other.seed_pair.target_position_in_read) && (seed_pair.query_position_in_read == other.seed_pair.query_position_in_read) && (length == other.length) && (score == other.score));
    }
};

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
    virtual StatusType extend_async(const char* h_query, const int32_t& query_length,
                                    const char* h_target, const int32_t& target_length,
                                    const int32_t& score_threshold,
                                    const std::vector<SeedPair>& h_seed_pairs) = 0;

    /// \brief Device pointer prototype for extension
    ///
    /// Memcopies to device memory are assumed to be done before this
    /// function. Output array d_scored_segment_pairs must be pre-allocated on device.
    /// Launches async extension kernel. Filters segments on device
    /// based on score_threshold.
    virtual StatusType extend_async(const char* d_query, int32_t query_length,
                                    const char* d_target, int32_t target_length,
                                    int32_t score_threshold, SeedPair* d_seed_pairs,
                                    int32_t num_seed_pairs, ScoredSegmentPair* d_scored_segment_pairs,
                                    int32_t* d_num_scored_segment_pairs) = 0;

    /// \brief Waits for CUDA accelerated extension to finish
    ///
    /// Blocking call that waits for all the extensions scheduled
    /// on the GPU to come to completion.
    virtual StatusType sync() = 0;

    /// \brief Return the computed segment pairs
    ///
    /// \return Vector of Scored Segment Pairs
    virtual const std::vector<ScoredSegmentPair>& get_scored_segment_pairs() const = 0;

    /// \brief Reset Extender object and free device/host memory
    virtual void reset() = 0;
};

std::unique_ptr<Extender> create_extender(const int32_t* h_sub_mat, const int32_t sub_mat_dim, const int32_t xdrop_threshold, const bool no_entropy, cudaStream_t stream, const int32_t device_id, DefaultDeviceAllocator allocator, const ExtensionType type = ExtensionType::ungapped_xdrop);

} // namespace cudaextender
} // namespace genomeworks
} // namespace claraparabricks
