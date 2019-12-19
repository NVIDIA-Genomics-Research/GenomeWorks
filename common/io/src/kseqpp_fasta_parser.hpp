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
            FastaParserKseqpp(const std::string &fasta_file);
            ~FastaParserKseqpp();

            int32_t get_num_seqences() const override;

            FastaSequence get_sequence_by_id(int32_t i) const override;

        private:
            std::vector<FastaSequence> reads_;
            int32_t num_seqequences_;
        };

    } // namespace io
} // namespace claragenomics
