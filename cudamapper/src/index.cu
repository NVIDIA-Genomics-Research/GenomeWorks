/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "claragenomics/cudamapper/index.hpp"
#include <claragenomics/utils/cudautils.hpp>
#include "index_gpu.cuh"
#include "minimizer.hpp"

namespace claragenomics
{
namespace cudamapper
{
std::unique_ptr<Index> Index::create_index(const std::vector<io::FastaParser*>& parsers,
                                           const std::uint64_t kmer_size,
                                           const std::uint64_t window_size,
                                           const std::vector<std::pair<std::uint64_t, std::uint64_t>>& read_ranges)
{
    CGA_NVTX_RANGE(profiler, "create_index");
    return std::make_unique<IndexGPU<Minimizer>>(parsers, kmer_size, window_size, read_ranges);
}

std::unique_ptr<Index> Index::create_index()
{
    return std::make_unique<IndexGPU<Minimizer>>();
}
} // namespace cudamapper
} // namespace claragenomics
