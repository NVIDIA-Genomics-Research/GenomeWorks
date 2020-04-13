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

#include "claragenomics/io/fasta_parser.hpp"

#include <string>
#include <vector>

namespace claragenomics
{
namespace io
{

class FastaParserKseqpp : public FastaParser
{
public:
    /// \brief Constructor
    /// \param fasta_file Path to FASTA(.gz) file. If .gz, it must be zipped with bgzip.
    /// \param min_sequence_length Minimum length a sequence needs to be to be parsed. Shorter sequences are ignored.
    /// \param shuffle Enables shuffling reads
    FastaParserKseqpp(const std::string& fasta_file,
                      const number_of_basepairs_t min_sequence_length,
                      const bool shuffle);

    /// \brief Return number of sequences in FASTA file
    /// \return Sequence count in file
    number_of_reads_t get_num_seqences() const override;

    /// \brief Fetch an entry from the FASTA file by index position in file.
    /// \param sequence_id Position of sequence in file. If sequence_id is invalid an error is thrown.
    /// \return A FastaSequence object describing the entry.
    FastaSequence get_sequence_by_id(const read_id_t sequence_id) const override;

    /// \brief returns a list of pairs of read_id values where each range has at most max_chunk_size basepairs
    /// If a single sequence exceeds max_chunk_size it will be placed in its own chunk.
    ///
    /// \param max_chunk_size the maximum number of basepairs in a chunk (range of indices)
    /// \return first and past-the-last read_id of each chunk
    std::vector<std::pair<read_id_t, read_id_t>> get_read_chunks(const number_of_basepairs_t max_chunk_size) const override;

private:
    /// All the reads from the FASTA file are stored in host RAM
    /// given a sufficiently-large FASTA file, there may not be enough host RAM
    /// on the system
    std::vector<FastaSequence> reads_;
    std::vector<std::pair<read_id_t, read_id_t>> read_chunks_;
};

} // namespace io
} // namespace claragenomics
