/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "cudamapper_utils.hpp"

#include <cassert>

#include <omp.h>

#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/cudautils.hpp>

namespace claragenomics
{
namespace cudamapper
{

void print_paf(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigar,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               const std::int32_t kmer_size,
               std::mutex& write_output_mutex)
{
    CGA_NVTX_RANGE(profiler, "print_paf");

    assert(!cigar.empty() || (overlaps.size() && cigar.size()));

#pragma omp parallel
    {
        std::size_t number_of_threads = omp_get_num_threads();
        // All overlaps are saved to a single vector of chars and that vector is then printed to output.
        // Writing overlaps directly to output would be inefficinet as all writes to output have to protected by a mutex.
        //
        // Allocate approximately 150 characters for each overlap which will be processed by this thread,
        // if more characters are needed buffer will be reallocated.
        std::vector<char> buffer(150 * overlaps.size() / number_of_threads);
        // characters written buffer so far
        std::int64_t chars_in_buffer = 0;

#pragma omp for
        for (std::size_t i = 0; i < overlaps.size(); ++i)
        {
            const std::string& query_read_name  = query_parser.get_sequence_by_id(overlaps[i].query_read_id_).name;
            const std::string& target_read_name = target_parser.get_sequence_by_id(overlaps[i].target_read_id_).name;

            // (over)estimate the number of character that are going to be needed
            // 150 is an overestimate of number of characters that are going to be needed for non-string values
            std::int32_t expected_chars = 150 + query_read_name.length() + target_read_name.length();
            if (!cigar.empty())
            {
                expected_chars += cigar[i].length();
            }

            // if there is not enough space in buffer reallocate
            if (buffer.size() - chars_in_buffer < expected_chars)
            {
                buffer.resize(buffer.size() * 2 + expected_chars);
            }

            // Add basic overlap information.
            std::int32_t added_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                                    "%s\t%lu\t%i\t%i\t%c\t%s\t%lu\t%i\t%i\t%i\t%ldem\t%i",
                                                    query_read_name.c_str(),
                                                    query_parser.get_sequence_by_id(overlaps[i].query_read_id_).seq.length(),
                                                    overlaps[i].query_start_position_in_read_,
                                                    overlaps[i].query_end_position_in_read_,
                                                    static_cast<unsigned char>(overlaps[i].relative_strand),
                                                    target_read_name.c_str(),
                                                    target_parser.get_sequence_by_id(overlaps[i].target_read_id_).seq.length(),
                                                    overlaps[i].target_start_position_in_read_,
                                                    overlaps[i].target_end_position_in_read_,
                                                    overlaps[i].num_residues_ * kmer_size, // Print out the number of residue matches multiplied by kmer size to get approximate number of matching bases
                                                    std::max(std::abs(static_cast<std::int64_t>(overlaps[i].target_start_position_in_read_) - static_cast<std::int64_t>(overlaps[i].target_end_position_in_read_)),
                                                             std::abs(static_cast<std::int64_t>(overlaps[i].query_start_position_in_read_) - static_cast<std::int64_t>(overlaps[i].query_end_position_in_read_))), //Approximate alignment length
                                                    255);
            chars_in_buffer += added_chars;

            // If CIGAR string is generated, output in PAF.
            if (!cigar.empty())
            {
                added_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                           "\tcg:Z:%s",
                                           cigar[i].c_str());
                chars_in_buffer += added_chars;
            }

            // Add new line to demarcate new entry.
            buffer.data()[chars_in_buffer] = '\n';
            ++chars_in_buffer;
        }
        buffer[chars_in_buffer] = '\0';

        {
            CGA_NVTX_RANGE(profiler, "print_paf::writing_to_disk");
            std::lock_guard<std::mutex> lg(write_output_mutex);
            printf("%s", buffer.data());
        }
    }
}

DefaultDeviceAllocator get_device_allocator(const std::size_t max_cached_memory_bytes)
{
#ifdef CGA_ENABLE_CACHING_ALLOCATOR
    return {max_cached_memory_bytes};
#else
    // uses CudaMallocAllocator
    return {};
#endif
}

} // namespace cudamapper
} // namespace claragenomics
