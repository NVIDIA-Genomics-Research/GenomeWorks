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
#include "host_cache.cuh"
#include "index_gpu.cuh"
#include "minimizer.hpp"

namespace claragenomics
{
namespace cudamapper
{

HostCache::HostCache(const Index& index,
                     const read_id_t first_read_id,
                     const std::uint64_t kmer_size,
                     const std::uint64_t window_size)
    : first_read_id_(first_read_id)
    , kmer_size_(kmer_size)
    , window_size_(window_size)
{
    CGA_NVTX_RANGE(profiler, "cache_index");

    representations_.resize(index.representations().size());
    cudautils::device_copy_n(index.representations().data(), index.representations().size(), representations_.data());

    read_ids_.resize(index.read_ids().size());
    cudautils::device_copy_n(index.read_ids().data(), index.read_ids().size(), read_ids_.data());

    positions_in_reads_.resize(index.positions_in_reads().size());
    cudautils::device_copy_n(index.positions_in_reads().data(), index.positions_in_reads().size(), positions_in_reads_.data());

    directions_of_reads_.resize(index.directions_of_reads().size());
    cudautils::device_copy_n(index.directions_of_reads().data(), index.directions_of_reads().size(), directions_of_reads_.data());

    unique_representations_.resize(index.unique_representations().size());
    cudautils::device_copy_n(index.unique_representations().data(), index.unique_representations().size(), unique_representations_.data());

    first_occurrence_of_representations_.resize(index.first_occurrence_of_representations().size());
    cudautils::device_copy_n(index.first_occurrence_of_representations().data(), index.first_occurrence_of_representations().size(), first_occurrence_of_representations_.data());

    read_id_to_read_name_.resize(index.read_ids_to_read_names().size());
    thrust::copy(index.read_ids_to_read_names().begin(), index.read_ids_to_read_names().end(), read_id_to_read_name_.begin()); //H2H, may replace with shared_ptr

    read_id_to_read_length_.resize(index.read_ids_to_read_lengths().size());
    thrust::copy(index.read_ids_to_read_lengths().begin(), index.read_ids_to_read_lengths().end(), read_id_to_read_length_.begin()); //H2H, may replace with shared_ptr

    number_of_reads_                     = index.number_of_reads();
    number_of_basepairs_in_longest_read_ = index.number_of_basepairs_in_longest_read();
}

std::unique_ptr<Index> HostCache::copy_index_to_device(std::shared_ptr<claragenomics::DeviceAllocator> allocator)
{
    return std::make_unique<IndexGPU<Minimizer>>(allocator, *this);
}

const std::vector<representation_t>& HostCache::representations() const
{
    return representations_;
}

const std::vector<read_id_t>& HostCache::read_ids() const
{
    return read_ids_;
}

const std::vector<position_in_read_t>& HostCache::positions_in_reads() const
{
    return positions_in_reads_;
}

const std::vector<SketchElement::DirectionOfRepresentation>& HostCache::directions_of_reads() const
{
    return directions_of_reads_;
}

const std::vector<representation_t>& HostCache::unique_representations() const
{
    return unique_representations_;
}

const std::vector<std::uint32_t>& HostCache::first_occurrence_of_representations() const
{
    return first_occurrence_of_representations_;
}

const std::vector<std::string>& HostCache::read_id_to_read_names() const
{
    return read_id_to_read_name_;
}

const std::vector<std::uint32_t>& HostCache::read_id_to_read_lengths() const
{
    return read_id_to_read_length_;
}

read_id_t HostCache::number_of_reads() const
{
    return number_of_reads_;
}

position_in_read_t HostCache::number_of_basepairs_in_longest_read() const
{
    return number_of_basepairs_in_longest_read_;
}

read_id_t HostCache::first_read_id() const
{
    return first_read_id_;
}

std::uint64_t HostCache::kmer_size() const
{
    return kmer_size_;
}

std::uint64_t HostCache::window_size() const
{
    return window_size_;
}

} // namespace cudamapper
} // namespace claragenomics