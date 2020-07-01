/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <thrust/copy.h>
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

namespace details
{

/// IndexHostMemoryPinner - registers all host arrays in given IndexHostCopy as pinned memory and unregisters it at the end
class IndexHostMemoryPinner
{
public:
    /// @brief Constructor - registers pinned memory
    /// @param index_host_copy - IndexHostCopy whose arrays should be registered
    IndexHostMemoryPinner(IndexHostCopy& index_host_copy)
        : index_host_copy_(index_host_copy)
    {
        GW_CU_CHECK_ERR(cudaHostRegister(index_host_copy_.underlying_array_.data(),
                                         index_host_copy_.underlying_array_.size() * sizeof(unsigned char),
                                         cudaHostRegisterDefault));
    }

    /// @brief Destructor - unregisters the arrays
    ~IndexHostMemoryPinner()
    {
        GW_CU_CHECK_ERR(cudaHostUnregister(index_host_copy_.underlying_array_.data()));
    }

private:
    IndexHostCopy& index_host_copy_;
};

} // namespace details

IndexHostCopy::IndexHostCopy(const Index& index,
                             const read_id_t first_read_id,
                             const std::uint64_t kmer_size,
                             const std::uint64_t window_size,
                             const cudaStream_t cuda_stream)
    : first_read_id_(first_read_id)
    , kmer_size_(kmer_size)
    , window_size_(window_size)
{
    GW_NVTX_RANGE(profiler, "cache_index");

    // Use only one large array to store all arrays in order to reduce fragmentation when using pool allocators
    // Align all arrays by 64 bits
    const std::size_t representations_bits                     = claraparabricks::genomeworks::ceiling_divide(index.representations().size() * sizeof(representation_t), sizeof(std::uint64_t)) * sizeof(uint64_t);
    const std::size_t read_ids_bits                            = claraparabricks::genomeworks::ceiling_divide(index.read_ids().size() * sizeof(read_id_t), sizeof(std::uint64_t)) * sizeof(uint64_t);
    const std::size_t positions_in_reads_bits                  = claraparabricks::genomeworks::ceiling_divide(index.positions_in_reads().size() * sizeof(position_in_read_t), sizeof(std::uint64_t)) * sizeof(uint64_t);
    const std::size_t directions_of_reads_bits                 = claraparabricks::genomeworks::ceiling_divide(index.directions_of_reads().size() * sizeof(SketchElement::DirectionOfRepresentation), sizeof(std::uint64_t)) * sizeof(uint64_t);
    const std::size_t unique_representations_bits              = claraparabricks::genomeworks::ceiling_divide(index.unique_representations().size() * sizeof(representation_t), sizeof(std::uint64_t)) * sizeof(uint64_t);
    const std::size_t first_occurrence_of_representations_bits = claraparabricks::genomeworks::ceiling_divide(index.first_occurrence_of_representations().size() * sizeof(std::uint32_t), sizeof(std::uint64_t)) * sizeof(uint64_t);

    const std::size_t total_bits = representations_bits +
                                   read_ids_bits +
                                   positions_in_reads_bits +
                                   directions_of_reads_bits +
                                   unique_representations_bits +
                                   first_occurrence_of_representations_bits;
    underlying_array_.resize(total_bits);

    std::size_t current_bit = 0;
    representations_ = {reinterpret_cast<representation_t*>(underlying_array_.data() + current_bit), index.representations().size()};
    current_bit += representations_bits;
    read_ids_ = {reinterpret_cast<read_id_t*>(underlying_array_.data() + current_bit), index.read_ids().size()};
    current_bit += read_ids_bits;
    positions_in_reads_ = {reinterpret_cast<position_in_read_t*>(underlying_array_.data() + current_bit), index.positions_in_reads().size()};
    current_bit += positions_in_reads_bits;
    directions_of_reads_ = {reinterpret_cast<SketchElement::DirectionOfRepresentation*>(underlying_array_.data() + current_bit), index.directions_of_reads().size()};
    current_bit += directions_of_reads_bits;
    unique_representations_ = {reinterpret_cast<representation_t*>(underlying_array_.data() + current_bit), index.unique_representations().size()};
    current_bit += unique_representations_bits;
    first_occurrence_of_representations_ = {reinterpret_cast<std::uint32_t*>(underlying_array_.data() + current_bit), index.first_occurrence_of_representations().size()};

    cudautils::device_copy_n(index.representations().data(),
                             index.representations().size(),
                             representations_.data,
                             cuda_stream);

    cudautils::device_copy_n(index.read_ids().data(),
                             index.read_ids().size(),
                             read_ids_.data,
                             cuda_stream);

    cudautils::device_copy_n(index.positions_in_reads().data(),
                             index.positions_in_reads().size(),
                             positions_in_reads_.data,
                             cuda_stream);

    cudautils::device_copy_n(index.directions_of_reads().data(),
                             index.directions_of_reads().size(),
                             directions_of_reads_.data,
                             cuda_stream);

    cudautils::device_copy_n(index.unique_representations().data(),
                             index.unique_representations().size(),
                             unique_representations_.data,
                             cuda_stream);
;
    cudautils::device_copy_n(index.first_occurrence_of_representations().data(),
                             index.first_occurrence_of_representations().size(),
                             first_occurrence_of_representations_.data,
                             cuda_stream);

    number_of_reads_                     = index.number_of_reads();
    number_of_basepairs_in_longest_read_ = index.number_of_basepairs_in_longest_read();

    // This is not completely necessary, but if removed one has to make sure that the next step
    // uses the same stream or that sync is done in caller
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
}

std::unique_ptr<Index> IndexHostCopy::copy_index_to_device(DefaultDeviceAllocator allocator,
                                                           const cudaStream_t cuda_stream) const
{
    // pin_memory_object registers all host arrays as pinned memory and unregisters them on its destruction (i.e. at the end of this function)
    details::IndexHostMemoryPinner pin_memory_object(const_cast<IndexHostCopy&>(*this));

    return std::make_unique<IndexGPU<Minimizer>>(allocator,
                                                 *this,
                                                 cuda_stream);
}

const IndexHostCopyBase::ArrayView<representation_t> IndexHostCopy::representations() const
{
    return representations_;
}

const IndexHostCopyBase::ArrayView<read_id_t> IndexHostCopy::read_ids() const
{
    return read_ids_;
}

const IndexHostCopyBase::ArrayView<position_in_read_t> IndexHostCopy::positions_in_reads() const
{
    return positions_in_reads_;
}

const IndexHostCopyBase::ArrayView<SketchElement::DirectionOfRepresentation> IndexHostCopy::directions_of_reads() const
{
    return directions_of_reads_;
}

const IndexHostCopyBase::ArrayView<representation_t> IndexHostCopy::unique_representations() const
{
    return unique_representations_;
}

const IndexHostCopyBase::ArrayView<std::uint32_t> IndexHostCopy::first_occurrence_of_representations() const
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

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
