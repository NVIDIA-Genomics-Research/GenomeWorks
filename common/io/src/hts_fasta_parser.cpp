/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "hts_fasta_parser.hpp"

#include <string>
#include <memory>

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

FastaParserHTS::FastaParserHTS(const std::string& fasta_file)
{
    fasta_index_ = fai_load3(fasta_file.c_str(), NULL, NULL, FAI_CREATE);
    if (fasta_index_ == NULL)
    {
        throw std::runtime_error("Could not load fasta index!");
    }

    num_seqequences_ = faidx_nseq(fasta_index_);
    if (num_seqequences_ == 0)
    {
        fai_destroy(fasta_index_);
        throw std::runtime_error("FASTA file has 0 sequences");
    }
}

FastaParserHTS::~FastaParserHTS()
{
    fai_destroy(fasta_index_);
}

int32_t FastaParserHTS::get_num_seqences() const
{
    return num_seqequences_;
}

FastaSequence FastaParserHTS::get_sequence_by_id(int32_t i) const
{
    std::string str_name = "";
    {
        std::lock_guard<std::mutex> lock(index_mutex_);
        const char* name = faidx_iseq(fasta_index_, i);
        if (name == NULL)
        {
            throw std::runtime_error("No sequence found for ID " + std::to_string(i));
        }
        str_name = std::string(name);
    }

    return get_sequence_by_name(str_name);
}

FastaSequence FastaParserHTS::get_sequence_by_name(const std::string& name) const
{

    FastaSequence s{};
    {
        std::lock_guard<std::mutex> lock(index_mutex_);
        int32_t length;
        std::unique_ptr<char, free_deleter> seq(fai_fetch(fasta_index_, name.c_str(), &length));
        if (length < 0)
        {
            throw std::runtime_error("Error in reading sequence information for seq ID " + name);
        }
        s.name = std::string(name);
        s.seq  = std::string(seq.get());
    }

    return s;
}

std::vector<std::pair<int, int>> FastaParserHTS::get_read_chunks(int max_chunk_size) const
{
    std::vector<std::pair<int, int>> chunks;

    std::pair<int, int> chunk;

    chunk.first   = 0;
    int num_bases = 0;
    for (int read_idx = 0; read_idx < num_seqequences_; read_idx++)
    {
        if (get_sequence_by_id(read_idx).seq.size() + num_bases > max_chunk_size)
        {
            chunk.second = read_idx - 1;
            chunks.push_back(chunk);
            chunk.first = read_idx;
            num_bases   = get_sequence_by_id(read_idx).seq.size();
        }
        else
        {
            num_bases += get_sequence_by_id(read_idx).seq.size();
        }
    }

    chunk.second = num_seqequences_ - 1;

    chunks.push_back(chunk);
    return chunks;
}

} // namespace io
} // namespace claragenomics
