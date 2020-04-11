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

FastaParserKseqpp::FastaParserKseqpp(const std::string& fasta_file, int min_sequencece_length, bool shuffle)
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
        FastaSequence seq   = {record.name, record.seq};
        int sequence_length = get_size<int>(record.seq);
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

int32_t FastaParserKseqpp::get_num_seqences() const
{
    return reads_.size();
}

FastaSequence FastaParserKseqpp::get_sequence_by_id(int32_t i) const
{
    return reads_[i];
}

std::vector<std::pair<int, int>> FastaParserKseqpp::get_read_chunks(int max_chunk_size = 1000000) const
{
    std::vector<std::pair<int, int>> chunks;

    std::pair<int, int> chunk;

    const int n_reads = get_size<int>(reads_);
    chunk.first       = 0;
    int num_bases     = 0;
    for (int read_idx = 0; read_idx < n_reads; read_idx++)
    {
        if (get_size<int>(reads_[read_idx].seq) + num_bases > max_chunk_size)
        {
            chunk.second = read_idx;
            chunks.push_back(chunk);
            chunk.first = read_idx;
            num_bases   = get_size<int>(reads_[read_idx].seq);
        }
        else
        {
            num_bases += get_size<int>(reads_[read_idx].seq);
        }
    }

    chunk.second = get_size(reads_);

    chunks.push_back(chunk);
    return chunks;
}

} // namespace io
} // namespace claragenomics
