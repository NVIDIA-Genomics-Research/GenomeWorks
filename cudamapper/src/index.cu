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

std::unique_ptr<Index> Index::create_index(DefaultDeviceAllocator allocator,
                                           const io::FastaParser& parser,
                                           const read_id_t first_read_id,
                                           const read_id_t past_the_last_read_id,
                                           const std::uint64_t kmer_size,
                                           const std::uint64_t window_size,
                                           const bool hash_representations,
                                           const double filtering_parameter,
                                           const cudaStream_t cuda_stream)
{
    CGA_NVTX_RANGE(profiler, "create_index");
    return std::make_unique<IndexGPU<Minimizer>>(allocator,
                                                 parser,
                                                 first_read_id,
                                                 past_the_last_read_id,
                                                 kmer_size,
                                                 window_size,
                                                 hash_representations,
                                                 filtering_parameter,
                                                 cuda_stream);
}

std::unique_ptr<IndexHostCopyBase> IndexHostCopyBase::create_cache(const Index& index,
                                                                   const read_id_t first_read_id,
                                                                   const std::uint64_t kmer_size,
                                                                   const std::uint64_t window_size,
                                                                   const cudaStream_t cuda_stream)
{
    CGA_NVTX_RANGE(profiler, "cache_D2H");
    return std::make_unique<IndexHostCopy>(index,
                                           first_read_id,
                                           kmer_size,
                                           window_size,
                                           cuda_stream);
}

} // namespace cudamapper
} // namespace claragenomics