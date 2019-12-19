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

#include <string>
#include <memory>

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

        FastaParserKseqpp::FastaParserKseqpp(const std::string &fasta_file)
        {
            klibpp::KSeq record;
            klibpp::SeqStreamIn iss(fasta_file.data());
            std::vector<FastaSequence> seqs; //temp vector
            int total_len = 0;
            while (iss >> record) {
                FastaSequence seq = {record.name, record.seq};
                total_len += record.seq.size();
                reads_.push_back(seq);
            }
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

   } // namespace io
} // namespace claragenomics
