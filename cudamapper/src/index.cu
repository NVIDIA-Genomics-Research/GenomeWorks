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

std::unique_ptr<IndexHostCopy> IndexHostCopy::create_cache(const Index& index,
                                                           const read_id_t first_read_id,
                                                           const std::uint64_t kmer_size,
                                                           const std::uint64_t window_size)
{
    CGA_NVTX_RANGE(profiler, "cache_D2H");
    return std::make_unique<HostCache>(index, first_read_id, kmer_size, window_size);
}

} // namespace cudamapper
} // namespace claragenomics