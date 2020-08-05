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

#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

#include <memory>

namespace claraparabricks
{

namespace genomeworks
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

} // namespace genomeworks

} // namespace claraparabricks
