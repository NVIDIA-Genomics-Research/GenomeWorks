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

FastaSequence FastaParserKseqpp::get_sequence_by_id(const read_id_t sequence_id) const
{
    return reads_[sequence_id];
}

std::vector<IndexDescriptor> FastaParserKseqpp::get_index_descriptors(const number_of_basepairs_t max_index_size = 1000000) const
{
    std::vector<IndexDescriptor> index_descriptors;

    const number_of_reads_t n_reads                = get_size<number_of_reads_t>(reads_);
    read_id_t first_sequence_in_index              = 0;
    number_of_reads_t number_of_sequences_in_index = 0;
    number_of_basepairs_t num_bases                = 0;
    for (read_id_t read_idx = 0; read_idx < n_reads; read_idx++)
    {
        if (get_size<number_of_basepairs_t>(reads_[read_idx].seq) + num_bases > max_index_size)
        {
            // adding this sequence would lead to index_descriptor being larger than max_index_size
            // save current index_descriptor and start a new one
            index_descriptors.push_back({first_sequence_in_index, number_of_sequences_in_index});
            first_sequence_in_index      = read_idx;
            number_of_sequences_in_index = 1;
            num_bases                    = get_size<number_of_basepairs_t>(reads_[read_idx].seq);
        }
        else
        {
            // add this sequence to the current index_descriptor
            num_bases += get_size<number_of_basepairs_t>(reads_[read_idx].seq);
            ++number_of_sequences_in_index;
        }
    }

    // save last index_descriptor
    index_descriptors.push_back({first_sequence_in_index, number_of_sequences_in_index});

    return index_descriptors;
}

} // namespace io
} // namespace claragenomics
