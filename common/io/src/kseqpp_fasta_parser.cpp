/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "kseqpp_fasta_parser.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <exception>
#include <iostream>
#include "seqio.h" //TODO add this to 3rdparty
#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claragenomics
{
namespace io
{

FastaParserKseqpp::FastaParserKseqpp(const std::string& fasta_file,
                                     const number_of_basepairs_t min_sequencece_length,
                                     const bool shuffle)
{
    klibpp::KSeq record;
    klibpp::SeqStreamIn iss(fasta_file.data());

    std::vector<FastaSequence> seqs;

    iss >> record;
    if (iss.fail())
    {
        throw std::invalid_argument("Error: "
                                    "non-existent or empty file " +
                                    fasta_file + " !");
    }

    do
    {
        FastaSequence seq                     = {record.name, record.seq};
        number_of_basepairs_t sequence_length = get_size<number_of_basepairs_t>(record.seq);
        if (sequence_length >= min_sequencece_length)
        {
            reads_.push_back(std::move(seq));
        }
    } while (iss >> record);

    //For many applications, such as cudamapper, performance is better if reads are shuffled.
    if (shuffle)
    {
        std::mt19937 g(0); // seed for deterministic behaviour
        std::shuffle(reads_.begin(), reads_.end(), g);
    }
}

number_of_reads_t FastaParserKseqpp::get_num_seqences() const
{
    return reads_.size();
}

FastaSequence FastaParserKseqpp::get_sequence_by_id(const read_id_t sequence_id) const
{
    return reads_[sequence_id];
}

std::vector<std::pair<read_id_t, read_id_t>> FastaParserKseqpp::get_read_chunks(const number_of_basepairs_t max_chunk_size = 1000000) const
{
    std::vector<std::pair<read_id_t, read_id_t>> chunks;

    std::pair<read_id_t, read_id_t> chunk;

    const number_of_reads_t n_reads = get_size<number_of_reads_t>(reads_);
    chunk.first                     = 0;
    number_of_basepairs_t num_bases = 0;
    for (read_id_t read_idx = 0; read_idx < n_reads; read_idx++)
    {
        if (get_size<number_of_basepairs_t>(reads_[read_idx].seq) + num_bases > max_chunk_size)
        {
            // adding this sequence would lead to chunk being larger than max_chunk_size
            // save current chunk and start a new one
            chunk.second = read_idx;
            chunks.push_back(chunk);
            chunk.first = read_idx;
            num_bases   = get_size<number_of_basepairs_t>(reads_[read_idx].seq);
        }
        else
        {
            // add this sequence to the current chunk
            num_bases += get_size<number_of_basepairs_t>(reads_[read_idx].seq);
        }
    }

    // save last chunk
    chunk.second = get_size(reads_);
    chunks.push_back(chunk);

    return chunks;
}

} // namespace io
} // namespace claragenomics
