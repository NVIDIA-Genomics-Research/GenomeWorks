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
#include <mutex>

extern "C" {
#include <htslib/faidx.h>
}

namespace claragenomics
{
namespace io
{

class FastaParserHTS : public FastaParser
{
public:
    FastaParserHTS(const std::string& fasta_file);
    ~FastaParserHTS();

    int32_t get_num_seqences() const override;

    FastaSequence get_sequence_by_id(int32_t i) const override;

    std::vector<std::pair<int,int>> get_read_chunks(int max_chunk_size) const override ;

private:
    faidx_t* fasta_index_;
    mutable std::mutex index_mutex_;
    int32_t num_seqequences_;

protected:
    /// \brief Fetch an entry from the FASTA file by name.
    /// \param name Name of the sequence in FASTA file. If there is no entry
    ///             by that name, an error is thrown.
    ///
    /// \return A FastaSequence object describing the entry.
    FastaSequence get_sequence_by_name(const std::string& name) const; //TODO push this back to the public API
};

} // namespace io
} // namespace claragenomics
