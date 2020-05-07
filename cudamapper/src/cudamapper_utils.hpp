/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <mutex>
#include <vector>

#include <claragenomics/cudamapper/types.hpp>
#include <claragenomics/utils/allocator.hpp>

namespace claragenomics
{

namespace io
{
class FastaParser;
}; // namespace io

namespace cudamapper
{

/// \brief prints overlaps to stdout in <a href="https://github.com/lh3/miniasm/blob/master/PAF.md">PAF format</a>
/// \param overlaps vector of overlap objects
/// \param cigar cigar strings
/// \param query_parser needed for read names and lenghts
/// \param target_parser needed for read names and lenghts
/// \param kmer_size minimizer kmer size
/// \param write_output_mutex mutex that enables exclusive access to output stream
void print_paf(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigar,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               std::int32_t kmer_size,
               std::mutex& write_output_mutex);

/// \brief crated a device allocator
/// \param max_cached_memory_bytes
/// \return device allocator
DefaultDeviceAllocator get_device_allocator(const std::size_t max_cached_memory_bytes);

} // namespace cudamapper
} // namespace claragenomics
