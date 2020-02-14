/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include "index_gpu.cuh"
#include "minimizer.hpp"

namespace claragenomics
{
namespace cudamapper
{

std::unique_ptr<Index> Index::create_index(std::shared_ptr<DeviceAllocator> allocator,
                                           const io::FastaParser& parser,
                                           const read_id_t first_read_id,
                                           const read_id_t past_the_last_read_id,
                                           const std::uint64_t kmer_size,
                                           const std::uint64_t window_size,
                                           const bool hash_representations,
                                           const double filtering_parameter)
{
    CGA_NVTX_RANGE(profiler, "create_index");
    return std::make_unique<IndexGPU<Minimizer>>(allocator,
                                                 parser,
                                                 first_read_id,
                                                 past_the_last_read_id,
                                                 kmer_size,
                                                 window_size,
                                                 hash_representations,
                                                 filtering_parameter);
}

} // namespace cudamapper
} // namespace claragenomics



namespace claragenomics
{
namespace cudamapper
{

    IndexCache::IndexCache(const Index& index,
                           const read_id_t first_read_id_in,
                           const std::uint64_t kmer_size_in,
                           const std::uint64_t window_size_in)
            : first_read_id(first_read_id_in),
              kmer_size(kmer_size_in),
              window_size(window_size_in)
    {
        CGA_NVTX_RANGE(profiler, "cache_index");
        {
            auto const & src = index.representations();
            auto const sz = src.size();
            representations_.resize(sz);
            cudautils::device_copy_n(src.data(), sz, representations_.data());
        }

        {
            auto const &src = index.read_ids();
            auto const sz = src.size();
            read_ids_.resize(src.size());
            cudautils::device_copy_n(src.data(), sz, read_ids_.data());
        }

        {
            auto const & src = index.positions_in_reads();
            auto const sz = src.size();
            positions_in_reads_.resize(sz);
            cudautils::device_copy_n(src.data(), sz, positions_in_reads_.data());
        }

        {
            auto const & src = index.directions_of_reads();
            auto const sz = src.size();
            directions_of_reads_.resize(sz);
            cudautils::device_copy_n(src.data(), sz, directions_of_reads_.data());
        }

        {
            auto const & src = index.unique_representations();
            auto const sz = src.size();
            unique_representations_.resize(sz);
            cudautils::device_copy_n(src.data(), sz, unique_representations_.data());
        }

        {
            auto const & src = index.first_occurrence_of_representations();
            auto const sz = src.size();
            first_occurrence_of_representations_.resize(sz);
            cudautils::device_copy_n(src.data(), sz, first_occurrence_of_representations_.data());
        }

        {
            auto const & src = index.read_id_to_read_name();
            read_id_to_read_name_.resize(src.size());
            thrust::copy(src.begin(), src.end(), read_id_to_read_name_.begin());    //H2H, may replace with shared_ptr
        }

        {
            auto const & src = index.read_id_to_read_length();
            read_id_to_read_length_.resize(src.size());
            thrust::copy(src.begin(), src.end(), read_id_to_read_length_.begin());  //H2H, may replace with shared_ptr
        }
    }

    std::unique_ptr<Index> IndexCache::copy_index_to_device(std::shared_ptr<claragenomics::DeviceAllocator> allocator)
    {
        return std::make_unique<IndexGPU<Minimizer>>(allocator, *this);
    }

    const std::vector<representation_t>& IndexCache::representations() const
    {
        return representations_;
    }

    const std::vector<read_id_t>& IndexCache::read_ids() const
    {
        return read_ids_;
    }

    const std::vector<position_in_read_t>& IndexCache::positions_in_reads() const
    {
        return positions_in_reads_;
    }

    const std::vector<SketchElement::DirectionOfRepresentation>& IndexCache::directions_of_reads() const
    {
        return directions_of_reads_;
    }

    const std::vector<representation_t>& IndexCache::unique_representations() const
    {
        return unique_representations_;
    }

    const std::vector<std::uint32_t>& IndexCache::first_occurrence_of_representations() const
    {
        return first_occurrence_of_representations_;
    }

    const std::vector<std::string>& IndexCache::read_id_to_read_name() const
    {
        return read_id_to_read_name_;
    }

    const std::vector<std::uint32_t>& IndexCache::read_id_to_read_length() const
    {
        return read_id_to_read_length_;
    }


} // namespace cudamapper
} // namespace claragenomics