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
#include <claragenomics/utils/filesystem.hpp>

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

FastaParserHTS::FastaParserHTS(const std::string& fasta_file,
                               const std::string& output_dir)
{
    std::string fai_file = fasta_file + ".fai";
    std::string gzi_file = fasta_file + ".gzi";
    if (output_dir.length() != 0)
    {
        // resolve idx and gzi file path
        if (!claragenomics::filesystem::dirExists(output_dir))
        {
            throw std::runtime_error("Output dir " + output_dir +
                                     " not found or not a directory!");
        }

        std::string file_name =
            claragenomics::filesystem::resolveFileName(fasta_file);
        fai_file = output_dir + "/" + file_name + ".fai";
        gzi_file = output_dir + "/" + file_name + ".gzi";
    }

    fasta_index_ = fai_load3(
        fasta_file.c_str(), fai_file.c_str(), gzi_file.c_str(),
        FAI_CREATE);

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

} // namespace io
} // namespace claragenomics
