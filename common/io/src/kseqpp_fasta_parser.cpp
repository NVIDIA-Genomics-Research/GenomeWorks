/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "kseqpp_fasta_parser.hpp"

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <exception>
#include <iostream>
#include "seqio.h" //TODO add this to 3rdparty
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

namespace claraparabricks
{

namespace genomeworks
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

} // namespace genomeworks

} // namespace claraparabricks
