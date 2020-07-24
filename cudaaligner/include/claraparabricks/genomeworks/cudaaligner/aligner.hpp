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

#pragma once

#include <claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp>
#include <claraparabricks/genomeworks/utils/allocator.hpp>

#include <memory>
#include <vector>
#include <cuda_runtime_api.h>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

// Forward declaration of Alignment class.
class Alignment;

/// \addtogroup cudaaligner
/// \{

/// \class Aligner
/// CUDA Alignment object
class Aligner
{
public:
    /// \brief Virtual destructor for Aligner.
    virtual ~Aligner() = default;

    /// \brief Launch CUDA accelerated alignment
    ///
    /// Perform alignment on all Alignment objects previously
    /// inserted. This is an async call, and returns before alignment
    /// is fully finished. To sync the alignments, refer to the
    /// sync_alignments() call;
    /// To
    virtual StatusType align_all() = 0;

    /// \brief Waits for CUDA accelerated alignment to finish
    ///
    /// Blocking call that waits for all the alignments scheduled
    /// on the GPU to come to completion.
    virtual StatusType sync_alignments() = 0;

    /// \brief Add new alignment object. Only strings with characters
    ///        from the alphabet [ACGT] are guaranteed to provide correct results.
    ///
    /// \param query Query string
    /// \param query_length  Query string length
    /// \param target Target string
    /// \param target_length Target string length
    /// \param reverse_complement_query Reverse complement the query string
    /// \param reverse_complement_target Reverse complement the target string
    virtual StatusType add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length,
                                     bool reverse_complement_query = false, bool reverse_complement_target = false) = 0;

    /// \brief Return the computed alignments.
    ///
    /// \return Vector of Alignments.
    virtual const std::vector<std::shared_ptr<Alignment>>& get_alignments() const = 0;

    /// \brief Reset aligner object.
    virtual void reset() = 0;
};

/// \brief Created Aligner object - DEPRECATED API
///
/// \param max_query_length Maximum length of query string
/// \param max_target_length Maximum length of target string
/// \param max_alignments Maximum number of alignments to be performed
/// \param type Type of aligner to construct
/// \param allocator Allocator to use for internal device memory allocations
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(int32_t max_query_length, int32_t max_target_length, int32_t max_alignments, AlignmentType type, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id);

/// \brief Created Aligner object - DEPRECATED API
///
/// \param max_query_length Maximum length of query string
/// \param max_target_length Maximum length of target string
/// \param max_alignments Maximum number of alignments to be performed
/// \param type Type of aligner to construct
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
/// \param max_device_memory_allocator_caching_size Maximum amount of device memory to use for cached memory allocations the cudaaligner instance. max_device_memory_allocator_caching_size = -1 (default) means all available device memory. This parameter is ignored if the SDK is compiled for non-caching allocators.
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(int32_t max_query_length, int32_t max_target_length, int32_t max_alignments, AlignmentType type, cudaStream_t stream, int32_t device_id, int64_t max_device_memory_allocator_caching_size = -1);

/// \brief Created Aligner object
///
/// \param type Type of aligner to construct
/// \param max_bandwidth Maximum bandwidth for the Ukkonen band
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
/// \param allocator Allocator to use for internal device memory allocations
/// \param max_device_memory Maximum amount of device memory to use from passed in allocator in bytes (-1 for all available memory)
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(AlignmentType type, int32_t max_bandwidth, cudaStream_t stream, int32_t device_id, DefaultDeviceAllocator allocator, int64_t max_device_memory);

/// \brief Created Aligner object
///
/// \param type Type of aligner to construct
/// \param max_bandwidth Maximum bandwidth for the Ukkonen band
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
/// \param max_device_memory Maximum amount of device memory used in bytes (-1 (default) for all available memory).
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(AlignmentType type, int32_t max_bandwidth, cudaStream_t stream, int32_t device_id, int64_t max_device_memory = -1);
/// \}
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
