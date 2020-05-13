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
#include <thread>

#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claraparabricks
{

namespace genomeworks
{
namespace cudamapper
{

namespace
{

/// \brief prints part of data passed to print_paf(), to be run in a separate thread
/// \param overlaps see print_paf()
/// \param cigar see print_paf()
/// \param query_parser see print_paf()
/// \param target_parser see print_paf()
/// \param target_parser see print_paf()
/// \param kmer_size see print_paf()
/// \param write_output_mutex see print_paf()
/// \param first_overlap_to_print index of first overlap from overlaps to print
/// \param number_of_overlaps_to_print number of overlaps to print
void print_part_of_data(const std::vector<Overlap>& overlaps,
                        const std::vector<std::string>& cigar,
                        const io::FastaParser& query_parser,
                        const io::FastaParser& target_parser,
                        const int32_t kmer_size,
                        std::mutex& write_output_mutex,
                        const int64_t first_overlap_to_print,
                        const int64_t number_of_overlaps_to_print)
{
    if (number_of_overlaps_to_print <= 0)
    {
        return;
    }

    // All overlaps are saved to a single vector of chars and that vector is then printed to output.
    // Writing overlaps directly to output would be inefficinet as all writes to output have to protected by a mutex.
    //
    // Allocate approximately 150 characters for each overlap which will be processed by this thread,
    // if more characters are needed buffer will be reallocated.
    std::vector<char> buffer(150 * number_of_overlaps_to_print);
    // characters written buffer so far
    int64_t chars_in_buffer = 0;

    for (int64_t i = first_overlap_to_print; i < first_overlap_to_print + number_of_overlaps_to_print; ++i)
    {
        const std::string& query_read_name  = query_parser.get_sequence_by_id(overlaps[i].query_read_id_).name;
        const std::string& target_read_name = target_parser.get_sequence_by_id(overlaps[i].target_read_id_).name;
        // (over)estimate the number of character that are going to be needed
        // 150 is an overestimate of number of characters that are going to be needed for non-string values
        int32_t expected_chars = 150 + get_size<int32_t>(query_read_name) + get_size<int32_t>(target_read_name);
        if (!cigar.empty())
        {
            expected_chars += get_size<int32_t>(cigar[i]);
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
                                                 std::max(std::abs(static_cast<std::int64_t>(overlaps[i].target_start_position_in_read_) - static_cast<std::int64_t>(overlaps[i].target_end_position_in_read_)),
                                                          std::abs(static_cast<std::int64_t>(overlaps[i].query_start_position_in_read_) - static_cast<std::int64_t>(overlaps[i].query_end_position_in_read_))), //Approximate alignment length
                                                 255);
        chars_in_buffer += added_chars;
        // If CIGAR string is generated, output in PAF.
        if (!cigar.empty())
        {
            const int32_t added_cigar_chars = std::sprintf(buffer.data() + chars_in_buffer,
                                                           "\tcg:Z:%s",
                                                           cigar[i].c_str());
            chars_in_buffer += added_cigar_chars;
        }
        // Add new line to demarcate new entry.
        buffer[chars_in_buffer] = '\n';
        ++chars_in_buffer;
    }
    buffer[chars_in_buffer] = '\0';

    std::lock_guard<std::mutex> lg(write_output_mutex);
    printf("%s", buffer.data());
}

} // namespace

void print_paf(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigar,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               const int32_t kmer_size,
               std::mutex& write_output_mutex,
               const int32_t number_of_devices)
{
    assert(!cigar.empty() || (overlaps.size() == cigar.size()));

    // divide the work into several threads

    int32_t number_of_threads   = std::thread::hardware_concurrency() / number_of_devices; // We could use a better heuristic here
    int64_t overlaps_per_thread = get_size<int64_t>(overlaps) / number_of_threads;

    std::vector<std::thread> threads;

    for (int32_t thread_id = 0; thread_id < number_of_threads; ++thread_id)
    {
        threads.emplace_back(print_part_of_data,
                             std::ref(overlaps),
                             std::ref(cigar),
                             std::ref(query_parser),
                             std::ref(target_parser),
                             kmer_size,
                             std::ref(write_output_mutex),
                             thread_id * overlaps_per_thread,
                             thread_id != number_of_threads - 1 ? overlaps_per_thread : get_size<int64_t>(overlaps) - thread_id * overlaps_per_thread); // last thread prints all remaining overlaps
    }

    for (std::thread& thread : threads)
    {
        thread.join();
    }
}

} // namespace cudamapper
} // namespace genomeworks

} // namespace claraparabricks
