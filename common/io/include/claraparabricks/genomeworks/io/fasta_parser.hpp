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

#include <string>
#include <memory>
#include <vector>

#include <claraparabricks/genomeworks/types.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace io
{

/// A structure to hold details of a single FASTA entry.
typedef struct
{
    /// Name of sequence.
    std::string name;
    /// Base pair representation of sequence.
    std::string seq;
} FastaSequence;

/// \class FastaParser
/// FASTA file parser
class FastaParser
{
public:
    /// \brief FastaParser implementations can have custom destructors, so delcare the abstract dtor as default.
    virtual ~FastaParser() = default;

    /// \brief Return number of sequences in FASTA file
    /// \return Sequence count in file
    virtual number_of_reads_t get_num_seqences() const = 0;

    /// \brief Fetch an entry from the FASTA file by index position in file.
    /// \param sequence_id Position of sequence in file. If sequence_id is invalid an error is thrown.
    /// \return A reference to FastaSequence describing the entry.
    virtual const FastaSequence& get_sequence_by_id(read_id_t sequence_id) const = 0;
};

/// \brief A builder function that returns a FASTA parser object which uses KSEQPP.
///
/// \param fasta_file Path to FASTA(.gz) file. If .gz, it must be zipped with bgzip.
/// \param min_sequence_length Minimum length a sequence needs to be to be parsed. Shorter sequences are ignored.
/// \param shuffle Enables shuffling reads
///
/// \return A unique pointer to a constructed parser object.
std::unique_ptr<FastaParser> create_kseq_fasta_parser(const std::string& fasta_file,
                                                      number_of_basepairs_t min_sequence_length = 0,
                                                      bool shuffle                              = true);

} // namespace io

} // namespace genomeworks

} // namespace claraparabricks
