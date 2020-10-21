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

#pragma once

#include <mutex>
#include <vector>
#include <string>

#ifdef GW_BUILD_HTSLIB
#include "sam.h"
#endif

#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include <claraparabricks/genomeworks/cudamapper/index.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace io
{
class FastaParser;
}; // namespace io

namespace cudamapper
{

// enum to determine output format
enum class OutputFormat
{
    PAF,
    SAM,
    BAM
};

/// \brief prints overlaps to stdout in <a href="https://github.com/lh3/miniasm/blob/master/PAF.md">PAF format</a>
/// \param overlaps vector of overlap objects
/// \param cigars CIGAR strings. Empty vector if none exist
/// \param query_parser needed for read names and lengths
/// \param target_parser needed for read names and lengths
/// \param kmer_size minimizer kmer size
/// \param write_output_mutex mutex that enables exclusive access to output stream
void print_paf(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigars,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               int32_t kmer_size,
               std::mutex& write_output_mutex);

#ifdef GW_BUILD_HTSLIB
/// \brief prints overlaps to stdout in <a href="https://samtools.github.io/hts-specs/SAMv1.pdf">BAM format</a>
/// \param overlaps vector of overlap objects
/// \param cigars CIGAR strings. Empty vector if none exist
/// \param query_parser needed for read names and lengths
/// \param target_parser needed for read names and lengths
/// \param format print in either BAM or SAM
/// \param write_output_mutex mutex that enables exclusive access to output stream
/// \param argc (optional) number of command line arguments used to generated the @PG CL sections
/// \param argv (optional) command line arguments used to generated the @PG CL sections
void print_sam(const std::vector<Overlap>& overlaps,
               const std::vector<std::string>& cigars,
               const io::FastaParser& query_parser,
               const io::FastaParser& target_parser,
               OutputFormat format,
               std::mutex& write_output_mutex,
               int argc     = -1,
               char* argv[] = nullptr);
#endif
/// \brief returns a vector of IndexDescriptors in which the sum of basepairs of all reads in one IndexDescriptor is at most max_basepairs_per_index
/// If a single read exceeds max_chunk_size it will be placed in its own IndexDescriptor.
///
/// \param parser parser to get the reads from
/// \param max_basepairs_per_index the maximum number of basepairs in an IndexDescriptor
/// \return vector of IndexDescriptors
std::vector<IndexDescriptor> group_reads_into_indices(const io::FastaParser& parser,
                                                      number_of_basepairs_t max_basepairs_per_index = 1000000);

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
