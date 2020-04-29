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

const FastaSequence& FastaParserKseqpp::get_sequence_by_id(const read_id_t sequence_id) const
{
    return reads_[sequence_id];
}

} // namespace io
} // namespace claragenomics
