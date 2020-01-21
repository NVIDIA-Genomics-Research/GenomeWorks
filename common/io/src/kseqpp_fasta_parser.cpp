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
#include <string>

#include <iostream>
#include "seqio.h" //TODO add this to 3rdparty
#include <zlib.h>

extern "C" {
#include <htslib/faidx.h>
}

namespace
{
struct free_deleter
{
    template <typename T>
    void operator()(T* x)
    {
        std::free(x);
    }
};
} // namespace

namespace claragenomics
{
namespace io
{

FastaParserKseqpp::FastaParserKseqpp(const std::string& fasta_file)
{
    klibpp::KSeq record;
    klibpp::SeqStreamIn iss(fasta_file.data());
    std::vector<FastaSequence> seqs;
    int total_len = 0;
    while (iss >> record)
    {
        FastaSequence seq = {record.name, record.seq};
        total_len += record.seq.size();
        reads_.push_back(seq);
    }
    std::random_shuffle(reads_.begin(), reads_.end());
}

FastaParserKseqpp::~FastaParserKseqpp()
{
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
            chunk.second = read_idx - 1;
            chunks.push_back(chunk);
            chunk.first = read_idx;
            num_bases   = reads_[read_idx].seq.size();
        }
        else
        {
            num_bases += reads_[read_idx].seq.size();
        }
    }

    chunk.second = reads_.size() - 1;

    chunks.push_back(chunk);
    return chunks;
}

} // namespace io
} // namespace claragenomics
