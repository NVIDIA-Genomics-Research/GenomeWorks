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

#include <algorithm>
#include <cassert>
#include <vector>

#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

void print_paf(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigars,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               const int32_t kmer_size,
               std::mutex& write_output_mutex)
{
    GW_NVTX_RANGE(profiler, "print_paf");

    assert(cigars.empty() || (overlaps.size() == cigars.size()));

    const int64_t number_of_overlaps_to_print = get_size<int64_t>(overlaps);

    if (number_of_overlaps_to_print <= 0)
    {
        return;
    }

    // All overlaps are saved to a single vector of chars and that vector is then printed to output.
    // Writing overlaps directly to output would be inefficinet as all writes to output have to protected by a mutex.

    // Allocate approximately 150 characters for each overlap which will be processed,
    // if more characters are needed buffer will be reallocated.
    std::vector<char> buffer(150 * number_of_overlaps_to_print);
    // characters written buffer so far
    int64_t chars_in_buffer = 0;

    {
        GW_NVTX_RANGE(profiler, "print_paf::formatting_output");
        for (int64_t i = 0; i < number_of_overlaps_to_print; ++i)
        {
            const std::string& query_read_name  = query_parser.get_sequence_by_id(overlaps[i].query_read_id_).name;
            const std::string& target_read_name = target_parser.get_sequence_by_id(overlaps[i].target_read_id_).name;
            // (over)estimate the number of character that are going to be needed
            // 150 is an overestimate of number of characters that are going to be needed for non-string values
            int32_t expected_chars = 150 + get_size<int32_t>(query_read_name) + get_size<int32_t>(target_read_name);
            if (!cigars.empty())
            {
                expected_chars += get_size<int32_t>(cigars[i]);
            }
            // if there is not enough space in buffer reallocate
            if (get_size<int64_t>(buffer) - chars_in_buffer < expected_chars)
            {
                buffer.resize(buffer.size() * 2 + expected_chars);
            }
            // Add basic overlap information.
            const int32_t added_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                                     "%s\t%lu\t%i\t%i\t%c\t%s\t%lu\t%i\t%i\t%i\t%ld\t%i",
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
                                                     std::max(std::abs(static_cast<int64_t>(overlaps[i].target_start_position_in_read_) - static_cast<int64_t>(overlaps[i].target_end_position_in_read_)),
                                                              std::abs(static_cast<int64_t>(overlaps[i].query_start_position_in_read_) - static_cast<int64_t>(overlaps[i].query_end_position_in_read_))), //Approximate alignment length
                                                     255);
            chars_in_buffer += added_chars;
            // If CIGAR strings were generated, output in PAF.
            if (!cigars.empty())
            {
                const int32_t added_cigars_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                                                "\tcg:Z:%s",
                                                                cigars[i].c_str());
                chars_in_buffer += added_cigars_chars;
            }
            // Add new line to demarcate new entry.
            buffer[chars_in_buffer] = '\n';
            ++chars_in_buffer;
        }
        buffer[chars_in_buffer] = '\0';
    }

    {
        GW_NVTX_RANGE(profiler, "print_paf::writing_to_disk");
        std::lock_guard<std::mutex> lg(write_output_mutex);
        printf("%s", buffer.data());
    }
}

std::vector<gw_string_view_t> split_into_kmers(const gw_string_view_t& s, const std::int32_t kmer_size, const std::int32_t stride)
{
    const std::size_t kmer_count = s.length() - kmer_size + 1;
    std::vector<gw_string_view_t> kmers;

    if (s.length() < kmer_size)
    {
        kmers.push_back(s);
        return kmers;
    }

    for (std::size_t i = 0; i < kmer_count; i += stride)
    {
        kmers.push_back(s.substr(i, i + kmer_size));
    }
    return kmers;
}

template <typename T>
std::size_t count_shared_elements(const std::vector<T>& a, const std::vector<T>& b)
{
    std::size_t a_index      = 0;
    std::size_t b_index      = 0;
    std::size_t shared_count = 0;

    while (a_index < a.size() && b_index < b.size())
    {
        if (a[a_index] == b[b_index])
        {
            ++shared_count;
            ++a_index;
            ++b_index;
        }
        else if (a[a_index] < b[b_index])
        {
            ++a_index;
        }
        else
        {
            ++b_index;
        }
    }
    return shared_count;
}

float sequence_jaccard_similarity(const gw_string_view_t& a, const gw_string_view_t& b, const std::int32_t kmer_size, const std::int32_t stride)
{
    std::vector<gw_string_view_t> a_kmers = split_into_kmers(a, kmer_size, stride);
    std::vector<gw_string_view_t> b_kmers = split_into_kmers(b, kmer_size, stride);
    std::sort(std::begin(a_kmers), std::end(a_kmers));
    std::sort(std::begin(b_kmers), std::end(b_kmers));

    const std::size_t shared_kmers = count_shared_elements(a_kmers, b_kmers);
    // Calculate the set union size of a and b
    std::size_t union_size = a_kmers.size() + b_kmers.size() - shared_kmers;
    return static_cast<float>(shared_kmers) / static_cast<float>(union_size);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
