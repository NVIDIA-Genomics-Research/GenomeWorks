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

#include "index_host_copy.cuh"
#include "index_gpu.cuh"
#include "minimizer.hpp"

#include <claraparabricks/genomeworks/utils/mathutils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

IndexHostCopy::IndexHostCopy(const Index& index,
                             const read_id_t first_read_id,
                             const std::uint64_t kmer_size,
                             const std::uint64_t window_size,
                             const cudaStream_t cuda_stream)
    : first_read_id_(first_read_id)
    , kmer_size_(kmer_size)
    , window_size_(window_size)
    , memory_pinner_(*this)
    , cuda_stream_(cuda_stream)
{
    GW_NVTX_RANGE(profiler, "index_host_copy::constructor");

    // Use only one large array to store all arrays in order to reduce fragmentation when using pool allocators
    // Align all arrays according to the largest type
    constexpr size_t alignment_bytes = std::max({alignof(representation_t), alignof(read_id_t), alignof(position_in_read_t), alignof(SketchElement::DirectionOfRepresentation), alignof(std::uint32_t)});

    const std::size_t representations_bytes                     = claraparabricks::genomeworks::ceiling_divide(index.representations().size() * sizeof(representation_t), alignment_bytes) * alignment_bytes;
    const std::size_t read_ids_bytes                            = claraparabricks::genomeworks::ceiling_divide(index.read_ids().size() * sizeof(read_id_t), alignment_bytes) * alignment_bytes;
    const std::size_t positions_in_reads_bytes                  = claraparabricks::genomeworks::ceiling_divide(index.positions_in_reads().size() * sizeof(position_in_read_t), alignment_bytes) * alignment_bytes;
    const std::size_t directions_of_reads_bytes                 = claraparabricks::genomeworks::ceiling_divide(index.directions_of_reads().size() * sizeof(SketchElement::DirectionOfRepresentation), alignment_bytes) * alignment_bytes;
    const std::size_t unique_representations_bytes              = claraparabricks::genomeworks::ceiling_divide(index.unique_representations().size() * sizeof(representation_t), alignment_bytes) * alignment_bytes;
    const std::size_t first_occurrence_of_representations_bytes = claraparabricks::genomeworks::ceiling_divide(index.first_occurrence_of_representations().size() * sizeof(std::uint32_t), alignment_bytes) * alignment_bytes;

    const std::size_t total_bytes = representations_bytes +
                                    read_ids_bytes +
                                    positions_in_reads_bytes +
                                    directions_of_reads_bytes +
                                    unique_representations_bytes +
                                    first_occurrence_of_representations_bytes;

    {
        GW_NVTX_RANGE(profiler, "index_host_copy::constructor::allocate_host_memory");
        underlying_array_.resize(total_bytes);
    }

    std::size_t current_byte = 0;
    representations_         = {reinterpret_cast<representation_t*>(underlying_array_.data() + current_byte), index.representations().size()};
    current_byte += representations_bytes;
    read_ids_ = {reinterpret_cast<read_id_t*>(underlying_array_.data() + current_byte), index.read_ids().size()};
    current_byte += read_ids_bytes;
    positions_in_reads_ = {reinterpret_cast<position_in_read_t*>(underlying_array_.data() + current_byte), index.positions_in_reads().size()};
    current_byte += positions_in_reads_bytes;
    directions_of_reads_ = {reinterpret_cast<SketchElement::DirectionOfRepresentation*>(underlying_array_.data() + current_byte), index.directions_of_reads().size()};
    current_byte += directions_of_reads_bytes;
    unique_representations_ = {reinterpret_cast<representation_t*>(underlying_array_.data() + current_byte), index.unique_representations().size()};
    current_byte += unique_representations_bytes;
    first_occurrence_of_representations_ = {reinterpret_cast<std::uint32_t*>(underlying_array_.data() + current_byte), index.first_occurrence_of_representations().size()};

    // register pinned memory, memory gets unpinned in finish_copying()
    memory_pinner_.register_pinned_memory();

    cudautils::device_copy_n(index.representations().data(),
                             index.representations().size(),
                             representations_.data,
                             cuda_stream_);

    cudautils::device_copy_n(index.read_ids().data(),
                             index.read_ids().size(),
                             read_ids_.data,
                             cuda_stream_);

    cudautils::device_copy_n(index.positions_in_reads().data(),
                             index.positions_in_reads().size(),
                             positions_in_reads_.data,
                             cuda_stream_);

    cudautils::device_copy_n(index.directions_of_reads().data(),
                             index.directions_of_reads().size(),
                             directions_of_reads_.data,
                             cuda_stream_);

    cudautils::device_copy_n(index.unique_representations().data(),
                             index.unique_representations().size(),
                             unique_representations_.data,
                             cuda_stream_);

    cudautils::device_copy_n(index.first_occurrence_of_representations().data(),
                             index.first_occurrence_of_representations().size(),
                             first_occurrence_of_representations_.data,
                             cuda_stream_);

    number_of_reads_                     = index.number_of_reads();
    number_of_basepairs_in_longest_read_ = index.number_of_basepairs_in_longest_read();

    // no stream synchronization, synchronization done in finish_copying()
}

void IndexHostCopy::finish_copying() const
{
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream_));
    memory_pinner_.unregister_pinned_memory();
}

std::unique_ptr<Index> IndexHostCopy::copy_index_to_device(DefaultDeviceAllocator allocator,
                                                           const cudaStream_t cuda_stream) const
{
    GW_NVTX_RANGE(profiler, "index_host_copy::copy_index_to_device");
    // register pinned memory, memory gets unpinned in finish_copying()
    memory_pinner_.register_pinned_memory();

    return std::make_unique<IndexGPU<Minimizer>>(allocator,
                                                 this,
                                                 cuda_stream);

    // no stream synchronization, synchronization done in finish_copying()
}

const IndexHostCopyBase::Span<representation_t> IndexHostCopy::representations() const
{
    return representations_;
}

const IndexHostCopyBase::Span<read_id_t> IndexHostCopy::read_ids() const
{
    return read_ids_;
}

const IndexHostCopyBase::Span<position_in_read_t> IndexHostCopy::positions_in_reads() const
{
    return positions_in_reads_;
}

const IndexHostCopyBase::Span<SketchElement::DirectionOfRepresentation> IndexHostCopy::directions_of_reads() const
{
    return directions_of_reads_;
}

const IndexHostCopyBase::Span<representation_t> IndexHostCopy::unique_representations() const
{
    return unique_representations_;
}

const IndexHostCopyBase::Span<std::uint32_t> IndexHostCopy::first_occurrence_of_representations() const
{
    return first_occurrence_of_representations_;
}

read_id_t IndexHostCopy::number_of_reads() const
{
    return number_of_reads_;
}

position_in_read_t IndexHostCopy::number_of_basepairs_in_longest_read() const
{
    return number_of_basepairs_in_longest_read_;
}

read_id_t IndexHostCopy::first_read_id() const
{
    return first_read_id_;
}

std::uint64_t IndexHostCopy::kmer_size() const
{
    return kmer_size_;
}

std::uint64_t IndexHostCopy::window_size() const
{
    return window_size_;
}

IndexHostCopy::IndexHostMemoryPinner::IndexHostMemoryPinner(IndexHostCopy& index_host_copy)
    : index_host_copy_(index_host_copy)
    , times_memory_pinned_(0)
{
}

IndexHostCopy::IndexHostMemoryPinner::~IndexHostMemoryPinner()
{
    // if memory was not unregistered (due to either a bug or an expection) unregister it
    if (times_memory_pinned_ != 0)
    {
        assert(!"memory should always be unregistered by unregister_pinned_memory()");
        GW_NVTX_RANGE(profiler, "index_host_memory_pinner::unregister_pinned_memory");
        GW_CU_CHECK_ERR(cudaHostUnregister(index_host_copy_.underlying_array_.data()));
    }
}

void IndexHostCopy::IndexHostMemoryPinner::register_pinned_memory()
{
    // only pin memory if it hasn't been pinned yet
    if (0 == times_memory_pinned_)
    {
        GW_NVTX_RANGE(profiler, "index_host_memory_pinner::register_pinned_memory");
        GW_CU_CHECK_ERR(cudaHostRegister(index_host_copy_.underlying_array_.data(),
                                         index_host_copy_.underlying_array_.size() * sizeof(gw_byte_t),
                                         cudaHostRegisterDefault));
    }
    ++times_memory_pinned_;
}

void IndexHostCopy::IndexHostMemoryPinner::unregister_pinned_memory()
{
    assert(times_memory_pinned_ > 0);
    // only unpin memory if this is the last unpinning
    if (1 == times_memory_pinned_)
    {
        GW_NVTX_RANGE(profiler, "index_host_memory_pinner::unregister_pinned_memory");
        GW_CU_CHECK_ERR(cudaHostUnregister(index_host_copy_.underlying_array_.data()));
    }
    --times_memory_pinned_;
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
