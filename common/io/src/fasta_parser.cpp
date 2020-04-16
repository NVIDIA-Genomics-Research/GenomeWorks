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

#include "claragenomics/io/fasta_parser.hpp"

#include <memory>

namespace claragenomics
{
namespace io
{

std::unique_ptr<FastaParser> create_kseq_fasta_parser(const std::string& fasta_file,
                                                      const number_of_basepairs_t min_sequence_length,
                                                      const bool shuffle)
{
    return std::make_unique<FastaParserKseqpp>(fasta_file,
                                               min_sequence_length,
                                               shuffle);
}

} // namespace io
} // namespace claragenomics
