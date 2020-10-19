/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <cassert>
#include <string>

#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/version.hpp>

#include <claraparabricks/genomeworks/cudamapper/utils.hpp>

#ifdef GW_BUILD_HTSLIB
#include "kroundup.h"
#endif

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

#ifdef GW_BUILD_HTSLIB
namespace
{

static inline void encode_cigar(bam1_t* const alignment, const std::string& cigar)
{
    uint32_t* bam_cigar_start = bam_get_cigar(alignment);
    memcpy(bam_cigar_start, cigar.c_str(), cigar.length());
    alignment->core.n_cigar = cigar.length();

    return;
}

static inline void encode_seq(bam1_t* const alignment, const std::string& seq)
{
    // This table was taken from bam_construct_seq() for base -> nibble encoding
    static const char L[256] =
        {
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 0, 15, 15,
            15, 1, 14, 2, 13, 15, 15, 4, 11, 15, 15, 12, 15, 3, 15, 15,
            15, 15, 5, 6, 8, 15, 7, 9, 15, 10, 15, 15, 15, 15, 15, 15,
            15, 1, 14, 2, 13, 15, 15, 4, 11, 15, 15, 12, 15, 3, 15, 15,
            15, 15, 5, 6, 8, 15, 7, 9, 15, 10, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};

    const size_t len = seq.length();

    // encode bases as nibbles and write them into the alignment struct 2 bases at a time
    // 4 high bits are first base, 4 low bits are second base
    uint8_t* bam_seq_start = bam_get_seq(alignment);
    int i                  = 0;
    for (i; (i + 1) < len; i += 2)
    {
        *bam_seq_start++ = (L[static_cast<unsigned char>(seq[i])] << 4) + L[static_cast<unsigned char>(seq[i + 1])];
    }
    // odd number of bases, so we have one left over
    if (i < len)
    {
        *bam_seq_start++ = L[static_cast<unsigned char>(seq[i])] << 4;
    }

    alignment->core.l_qseq = len;

    return;
}

static inline void encode_qual(bam1_t* alignment)
{
    // if htslib encounters 0xff at the start of the QUAL section, it writes default value (*)
    auto bam_qual_start = bam_get_qual(alignment);
    *bam_qual_start     = 0xff;
}

} // namespace

void print_sam(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigars,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               const OutputFormat format,
               std::mutex& write_output_mutex,
               int argc,
               char* argv[])
{
    GW_NVTX_RANGE(profiler, "print_sam::formatting_output");

    // Assume output is stdout, and assume SAM output by default
    // Rest of the add()/write() functions choose based on samFile format
    // use hts_close here directly, because sam_close is a macro for hts_close, which the compiler did not like
    using samFilePtr_t = std::unique_ptr<samFile, decltype(&hts_close)>;

    auto file   = (format == OutputFormat::SAM) ? samFilePtr_t(sam_open("-", "wh"), &hts_close) : samFilePtr_t(sam_open("-", "bwh"), &hts_close);
    auto header = std::unique_ptr<bam_hdr_t, decltype(&sam_hdr_destroy)>(sam_hdr_init(), &sam_hdr_destroy);
    int result  = 0;

    if (file == nullptr)
    {
        throw std::runtime_error("print_sam: cannot open file stream for writing");
    }

    const int64_t number_of_overlaps_to_print = get_size<int64_t>(overlaps);
    for (int64_t i = 0; i < number_of_overlaps_to_print; ++i)
    {
        const std::string& query_read_name  = query_parser.get_sequence_by_id(overlaps[i].query_read_id_).name;
        const std::string& target_read_name = target_parser.get_sequence_by_id(overlaps[i].target_read_id_).name;
        const std::string length            = std::to_string(target_parser.get_sequence_by_id(overlaps[i].target_read_id_).seq.length());

        result = sam_hdr_add_line(header.get(), "SQ", "SN", target_read_name.c_str(), "LN", length.c_str(),
                                  "CL", NULL);
        if (result < 0)
        {
            fprintf(stderr, "print_sam: could not add header value");
        }
    }

    // If we have the information to write out the CL section, do it
    if ((argc != -1) && (argv != nullptr))
    {
        auto command_line = std::unique_ptr<char[], decltype(&free)>(stringify_argv(argc, argv), &free);
        result            = sam_hdr_add_pg(header.get(), "cudamapper", "VN", claraparabricks_genomeworks_version().c_str(),
                                "CL", *command_line.get(), NULL);
    }
    else
    {
        result = sam_hdr_add_pg(header.get(), "cudamapper", "VN", claraparabricks_genomeworks_version().c_str(), NULL);
    }
    if (result < 0)
    {
        fprintf(stderr, "print_sam: could not add PG header line");
    }

    {
        result = sam_hdr_write(file.get(), header.get());
        if (result < 0)
        {
            fprintf(stderr, "print_sam: could not add PG header line");
        }
    }

    // write out mandatory fields for alignments
    for (int64_t i = 0; i < number_of_overlaps_to_print; ++i)
    {
        const std::string& query_read_name = query_parser.get_sequence_by_id(overlaps[i].query_read_id_).name;
        auto alignment                     = std::unique_ptr<bam1_t, decltype(&bam_destroy1)>(bam_init1(), &bam_destroy1);

        // write QNAME field
        bam_set_qname(alignment.get(), query_read_name.c_str());

        // At this point we need to handle mem allocs ourselves, find necessary data size
        size_t total_size = 0;
        if (!cigars.empty())
        {
            total_size = alignment->core.l_qname + alignment->core.l_extranul                                                         // len query name + how many padding '\0' chars
                         + cigars[i].length() + ((query_parser.get_sequence_by_id(overlaps[i].query_read_id_).seq.length() + 1) / 2); // sequence length and div by 2 because bases in the struct are encoded as nibbles
        }
        else
        {
            // case where we don't have CIGAR strings. See above calculation for details
            total_size = alignment->core.l_qname + alignment->core.l_extranul + ((query_parser.get_sequence_by_id(overlaps[i].query_read_id_).seq.length() + 1) / 2);
        }
        const size_t max_data = kroundup64(total_size);
        if (total_size >= alignment->m_data)
        {
            alignment->data = static_cast<uint8_t*>(realloc(alignment->data, max_data));
        }
        alignment->l_data = total_size;
        alignment->m_data = max_data;

        // write FLAG field
        alignment->core.flag = 0;

        // write POS, MAPQ
        // NOTE hardcode MAPQ to 255 like in print_paf, but SAM specification says we shouldn't do this
        alignment->core.pos  = static_cast<int32_t>(overlaps[i].query_start_position_in_read_);
        alignment->core.qual = 255;

        // write CIGAR
        if (!cigars.empty())
        {
            encode_cigar(alignment.get(), cigars[i]);
        }

        // write RNEXT, PNEXT, and TLEN
        // all left as blank/default for now

        // write SEQ
        encode_seq(alignment.get(), query_parser.get_sequence_by_id(overlaps[i].query_read_id_).seq);

        // QUAL is set to * by default
        encode_qual(alignment.get());

        // TODO: write AUX data if available

        std::lock_guard<std::mutex> lg(write_output_mutex);
        int out = sam_write1(file.get(), header.get(), alignment.get());
    }

    return;
}
#endif

std::vector<IndexDescriptor> group_reads_into_indices(const io::FastaParser& parser,
                                                      const number_of_basepairs_t max_basepairs_per_index)
{
    std::vector<IndexDescriptor> index_descriptors;

    const number_of_reads_t total_number_of_reads              = parser.get_num_seqences();
    read_id_t first_read_in_current_index                      = 0;
    number_of_reads_t number_of_reads_in_current_index         = 0;
    number_of_basepairs_t number_of_basepairs_in_current_index = 0;
    for (read_id_t read_id = 0; read_id < total_number_of_reads; read_id++)
    {
        number_of_basepairs_t basepairs_in_this_read = get_size<number_of_basepairs_t>(parser.get_sequence_by_id(read_id).seq);
        if (basepairs_in_this_read + number_of_basepairs_in_current_index > max_basepairs_per_index)
        {
            // adding this sequence would lead to index_descriptor being larger than max_basepairs_per_index
            // save current index_descriptor and start a new one
            index_descriptors.push_back({first_read_in_current_index, number_of_reads_in_current_index});
            first_read_in_current_index          = read_id;
            number_of_reads_in_current_index     = 1;
            number_of_basepairs_in_current_index = basepairs_in_this_read;
        }
        else
        {
            // add this sequence to the current index_descriptor
            number_of_basepairs_in_current_index += basepairs_in_this_read;
            ++number_of_reads_in_current_index;
        }
    }

    // save last index_descriptor
    index_descriptors.push_back({first_read_in_current_index, number_of_reads_in_current_index});

    return index_descriptors;
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
