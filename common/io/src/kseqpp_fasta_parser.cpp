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

#include <iostream>
#include "seqio.h" //TODO add this to 3rdparty

namespace claragenomics
{
namespace io
{

FastaParserKseqpp::FastaParserKseqpp(const std::string& fasta_file, int min_sequencece_length)
{
    klibpp::KSeq record;
    klibpp::SeqStreamIn iss(fasta_file.data());
    std::vector<FastaSequence> seqs;
    int total_len = 0;
    while (iss >> record)
    {
        FastaSequence seq = {record.name, record.seq};
        auto sequence_length = record.seq.size();
        if (sequence_length >= min_sequencece_length) {
            total_len += sequence_length;
            reads_.emplace_back(seq);
        }
    }

    //For many applications, such as cudamapper, performance is better if reads are shuffled.
    std::mt19937 g(0); // seed for deterministic behaviour
    std::shuffle(reads_.begin(), reads_.end(), g);
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

    chunk.first   = 0;
    int num_bases = 0;
    for (int read_idx = 0; read_idx < reads_.size(); read_idx++)
    {
        if (reads_[read_idx].seq.size() + num_bases > max_chunk_size)
        {
            chunk.second = read_idx;
            chunks.push_back(chunk);
            chunk.first = read_idx;
            num_bases   = reads_[read_idx].seq.size();
        }
        else
        {
            num_bases += reads_[read_idx].seq.size();
        }
    }

    chunk.second = reads_.size();

    chunks.push_back(chunk);
    return chunks;
}

} // namespace io
} // namespace claragenomics
