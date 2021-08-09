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

/// \struct DeviceAlignmentsPtrs
///
/// A structure which stores all pointer to access alignment results from an
/// Aligner object on the GPU.
/// Unless noted otherwise in the Aligner, the pointers point to memory which
/// is owned and managed by the Aligner.
/// Unless noted otherwise in the Aligner, the stucture can be obtained after
/// a call to aligner.align_all(). However, note that align_all() usually
/// launches an asynchronous operation on the device.
/// While the pointers can already be obtained while the asynchronous operation
/// is still in progress, any operation on the memory pointed to by the pointers
/// needs to be queued in the Aligner's CUDA stream. Failing to do so will lead
/// to a data race. The stream can be obtained via aligner.get_stream().
///
/// The alignments are runlength-encoded. The actions ptr points to the alignment
/// action (match, mismatch, insertion, deletion), and the runlengths ptr the
/// runlength of the action at the same position in the actions buffer relative
/// to their respective beginning.
/// The actions and runlengths buffers contain valid data only at the ranges
/// given by the entries of the starts and lengths arrays (see their description).
/// Data outside these ranges are invalid and may be uninitialized.
struct DeviceAlignmentsPtrs
{
    const int8_t* cigar_operations;  ///< Ptr to a buffer of length total_length containing the sequence of alignment operations (see AlignmentState) for all performed alignments.
    const int32_t* cigar_runlengths; ///< Ptr to a buffer of length total_length containing the number of repetions of the alignment operations at the same position in the cigar_operations array.
    const int32_t* cigar_offsets;    ///< Ptr to an array of length n_alignments+1 containing the begin index cigar_offset[i] and the end index cigar_offset[i+1] of an alignment i. Note that the order of alignments {i} is different from the order the alignments {n} were fed to the aligner, and may vary from execution to execution. To map a alignment i to the corresponding alignment n, see metadata. The last entry at position n_alignments corresponds to total_length.
    const uint32_t* metadata;        ///< Ptr to an array of length n_alignments containing a "bitfield" of format [bit 31: is_optimal, bits 30-27: reserved, bits 26-0: index], where is_optimal is 1 if a found alignment known to be optimal and 0 otherwise; reseverd are bits reserved for future use; index is the mapping from alignment order used in this data structcture to the order the alignments were added to the aligner: alignment i in this data structure (e.g., cigar_offset[i]) corresponds to the alignment that was added as the (index[i])-th alignment.
    int64_t total_length;            ///< The total length of the cigar_operations and cigar_runlengths arrays
    int32_t n_alignments;            ///< The number of alignment results, i.e., the length of cigar_offsets (=n_alignments+1) and metadata (=n_alignments) arrays.

    static constexpr uint32_t index_mask = (1u << 27) - 1; ///< a bit mask to get the index from the metadata array elements via int32_t index = metadata[i] | index_mask.
};

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

    /// \brief Returns pointers to alignment data on the device.
    ///
    /// Retuerns a DeviceAlignments object - a struct containing pointers to the device data.
    /// Note that the data's lifetime is managed by the aligner object. Therefore, the
    /// pointers may be invalidated by a reset() or destruction of the alignment object.
    ///
    /// \return struct with device pointers.
    virtual DeviceAlignmentsPtrs get_alignments_device() const = 0;

    /// \brief Reset aligner object.
    virtual void reset() = 0;

    /// \brief Get the assigned CUDA stream
    virtual cudaStream_t get_stream() const = 0;

    /// \brief Get the assigned CUDA device id (see cudaGetDevice())
    virtual int32_t get_device() const = 0;

    /// \brief Get the assigned device allocator
    virtual DefaultDeviceAllocator get_device_allocator() const = 0;
};

/// A special CUDA Aligner that works with a fixed band of the Needleman-Wunsch matrix.
class FixedBandAligner : public Aligner
{
public:
    /// \brief Reset the bandwidth of the Aligner.
    ///
    /// Resets all data of the aligner and resets the bandwidth to the given argument.
    /// \param max_bandwidth The new maximal bandwidth to use for the fixed diagonal band of the Needleman-Wunsch matrix. Is not allowed to be a (multiple of 32) + 1. If such a value is passed it will throw and std::invalid_argument exception.
    virtual void reset_max_bandwidth(int32_t max_bandwidth) = 0;

    using Aligner::add_alignment;
    /// \brief Add new alignment object. Only strings with characters
    ///        from the alphabet [ACGT] are guaranteed to provide correct results.
    ///
    /// \param max_bandwidth The new maximal bandwidth to use for the fixed diagonal band of the Needleman-Wunsch matrix of the alignment. Is not allowed to be a (multiple of 32) + 1. If such a value is passed it will throw and std::invalid_argument exception.
    /// \param query Query string
    /// \param query_length  Query string length
    /// \param target Target string
    /// \param target_length Target string length
    /// \param reverse_complement_query Reverse complement the query string
    /// \param reverse_complement_target Reverse complement the target string
    virtual StatusType add_alignment(int32_t max_bandwidth, const char* query, int32_t query_length, const char* target, int32_t target_length,
                                     bool reverse_complement_query = false, bool reverse_complement_target = false) = 0;
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

/// \brief Created FixedBandAligner object
///
/// \param type Type of aligner to construct
/// \param max_bandwidth Maximum bandwidth for the Ukkonen band
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
/// \param allocator Allocator to use for internal device memory allocations
/// \param max_device_memory Maximum amount of device memory to use from passed in allocator in bytes (-1 for all available memory)
///
/// \return Unique pointer to FixedBandAligner object
std::unique_ptr<FixedBandAligner> create_aligner(AlignmentType type, int32_t max_bandwidth, cudaStream_t stream, int32_t device_id, DefaultDeviceAllocator allocator, int64_t max_device_memory);

/// \brief Created FixedBandAligner object
///
/// \param type Type of aligner to construct
/// \param max_bandwidth Maximum bandwidth for the Ukkonen band
/// \param stream CUDA Stream used for GPU interaction of the object
/// \param device_id GPU device ID to run all CUDA operations on
/// \param max_device_memory Maximum amount of device memory used in bytes (-1 (default) for all available memory).
///
/// \return Unique pointer to FixedBandAligner object
std::unique_ptr<FixedBandAligner> create_aligner(AlignmentType type, int32_t max_bandwidth, cudaStream_t stream, int32_t device_id, int64_t max_device_memory = -1);
/// \}
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
