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

#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>

#include "aligner_global_hirschberg_myers.hpp"
#include "aligner_global_myers_banded.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

std::unique_ptr<Aligner> create_aligner(
    int32_t max_query_length, int32_t max_target_length,
    int32_t max_alignments, AlignmentType type,
    DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
{
    if (type == AlignmentType::global_alignment)
    {
        return std::make_unique<AlignerGlobalHirschbergMyers>(max_query_length, max_target_length, max_alignments, allocator, stream, device_id);
    }
    else
    {
        throw std::runtime_error("Aligner for specified type not implemented yet.");
    }
}

std::unique_ptr<Aligner> create_aligner(
    int32_t max_query_length, int32_t max_target_length,
    int32_t max_alignments, AlignmentType type,
    cudaStream_t stream, int32_t device_id, int64_t max_device_memory_allocator_caching_size)
{
    if (max_device_memory_allocator_caching_size < -1)
    {
        throw std::invalid_argument("max_device_memory_allocator_caching_size has to be either -1 (=all available GPU memory) or greater or equal than 0.");
    }
#ifdef GW_ENABLE_CACHING_ALLOCATOR
    // uses CachingDeviceAllocator
    if (max_device_memory_allocator_caching_size == -1)
    {
        max_device_memory_allocator_caching_size = genomeworks::cudautils::find_largest_contiguous_device_memory_section();
        if (max_device_memory_allocator_caching_size == 0)
        {
            throw std::runtime_error("No memory available for caching");
        }
    }
    genomeworks::DefaultDeviceAllocator allocator(max_device_memory_allocator_caching_size);
#else
    // uses CudaMallocAllocator
    genomeworks::DefaultDeviceAllocator allocator;
#endif
    return create_aligner(max_query_length, max_target_length, max_alignments, type, allocator, stream, device_id);
}

std::unique_ptr<Aligner> create_aligner(
    const AlignmentType type,
    const int32_t max_bandwidth,
    cudaStream_t stream,
    const int32_t device_id,
    DefaultDeviceAllocator allocator,
    const int64_t max_device_memory)
{
    if (type == AlignmentType::global_alignment)
    {
        return std::make_unique<AlignerGlobalMyersBanded>(max_device_memory, max_bandwidth, allocator, stream, device_id);
    }
    else
    {
        throw std::runtime_error("Aligner for specified type not implemented yet.");
    }
}

std::unique_ptr<Aligner> create_aligner(
    const AlignmentType type,
    const int32_t max_bandwidth,
    cudaStream_t stream,
    const int32_t device_id,
    int64_t max_device_memory)
{
    if (max_device_memory < -1)
    {
        throw std::invalid_argument("max_device_memory has to be either -1 (=all available GPU memory) or greater or equal than 0.");
    }
#ifdef GW_ENABLE_CACHING_ALLOCATOR
    // uses CachingDeviceAllocator
    if (max_device_memory == -1)
    {
        max_device_memory = claraparabricks::genomeworks::cudautils::find_largest_contiguous_device_memory_section();
        if (max_device_memory == 0)
        {
            throw std::runtime_error("No memory available for caching");
        }
    }
    claraparabricks::genomeworks::DefaultDeviceAllocator allocator(max_device_memory);
#else
    // uses CudaMallocAllocator
    claraparabricks::genomeworks::DefaultDeviceAllocator allocator;
#endif
    return create_aligner(type, max_bandwidth, stream, device_id, allocator, -1);
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
