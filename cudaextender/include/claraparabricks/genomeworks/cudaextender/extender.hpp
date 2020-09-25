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

/// \addtogroup cudaextender
/// \{

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
};

/// Equality operator for ScoredSegmentPair
__host__ __device__ inline bool operator==(const ScoredSegmentPair& x, const ScoredSegmentPair& y)
{
    return ((x.seed_pair.target_position_in_read == y.seed_pair.target_position_in_read) && (x.seed_pair.query_position_in_read == y.seed_pair.query_position_in_read) && (x.length == y.length) && (x.score == y.score));
}

/// CUDA Extender Class
class Extender
{
public:
    /// \brief Virtual destructor for Extender.
    virtual ~Extender() = default;

    /// \brief Host pointer API for extension
    ///
    /// Takes values from host data structures, copies them over to device,
    /// launches async extension kernels on specified stream. Filters
    /// segments on device based on score_threshold
    ///
    /// \param[in] h_query Host pointer to encoded query sequence
    /// \param[in] query_length Length of query sequence
    /// \param[in] h_target Host pointer to encoded target sequence
    /// \param[in] target_length Length of target sequence
    /// \param[in] score_threshold Score threshold for filtering extended segments
    /// \param[in] h_seed_pairs Vector of seed pairs mapping to query and target
    ///                         sequences
    /// \return Status of the async extension launch
    virtual StatusType extend_async(const int8_t* h_query, const int32_t& query_length,
                                    const int8_t* h_target, const int32_t& target_length,
                                    const int32_t& score_threshold,
                                    const std::vector<SeedPair>& h_seed_pairs) = 0;

    /// \brief Device pointer API for extension
    ///
    /// All inputs to this function are expected to be resident on the device
    /// before launch. Output array d_scored_segment_pairs must be pre-allocated on device.
    /// Launches async extension kernel. Filters segments on device based on score_threshold.
    /// The stream associated with this function call must be synchronized to before accessing
    /// any output parameters.
    ///
    /// \param[in] d_query Device pointer to encoded query sequence
    /// \param[in] query_length Length of query sequence
    /// \param[in] d_target Device pointer to encoded target sequence
    /// \param[in] target_length Length of target sequence
    /// \param[in] score_threshold Score threshold for filtering extended segments
    /// \param[in] d_seed_pairs Device pointer to array of seed pairs mapping between
    ///                         target and query sequences
    /// \param[in] num_seed_pairs Length of d_seed_pairs array
    /// \param[out] d_scored_segment_pairs Pointer to a pre-allocated device location for
    ///                                    storing extension output
    /// \param[out] d_num_scored_segment_pairs Pointer to pre-allocated device location for
    ///                                        storing length of extension output
    /// \return Status of the async extension launch
    virtual StatusType extend_async(const int8_t* d_query, int32_t query_length,
                                    const int8_t* d_target, int32_t target_length,
                                    int32_t score_threshold, SeedPair* d_seed_pairs,
                                    int32_t num_seed_pairs, ScoredSegmentPair* d_scored_segment_pairs,
                                    int32_t* d_num_scored_segment_pairs) = 0;

    /// \brief Waits for CUDA accelerated extension to finish
    ///
    /// To be used with the host pointer extend_async API.
    /// Blocking call that waits for all the extensions scheduled
    /// on the GPU to come to completion.
    ///
    /// \return Synchronization and memory copy status
    virtual StatusType sync() = 0;

    /// \brief Returns the computed scored segment pairs
    ///
    /// To be used with the host pointer extend_async API. sync() must
    /// be called after launching async extension and before calling this function
    /// or the returned results will not be valid.
    ///
    /// \return Vector of Scored Segment Pairs that have a score >= score_threshold
    virtual const std::vector<ScoredSegmentPair>& get_scored_segment_pairs() const = 0;

    /// \brief Reset Extender object and free device/host memory
    virtual void reset() = 0;
};

/// \brief Create Extender object
///
/// \param h_score_mat Host pointer to scoring matrix for use during extension
/// \param score_mat_dim Dimension of the scoring matrix
/// \param xdrop_threshold Threshold for performing X-Drop
/// \param no_entropy Flag indicating whether to use entropy during extension
/// \param stream CUDA Stream to be used with extension
/// \param device_id GPU to be used for extension
/// \param allocator DeviceAllocator to be used for allocating/freeing memory
/// \param type Type of extension to be performed
/// \return Unique pointer to Extender object.
std::unique_ptr<Extender> create_extender(const int32_t* h_score_mat,
                                          const int32_t score_mat_dim,
                                          const int32_t xdrop_threshold,
                                          const bool no_entropy,
                                          cudaStream_t stream,
                                          const int32_t device_id,
                                          DefaultDeviceAllocator allocator,
                                          const ExtensionType type = ExtensionType::ungapped_xdrop);
/// \}
} // namespace cudaextender
} // namespace genomeworks
} // namespace claraparabricks
