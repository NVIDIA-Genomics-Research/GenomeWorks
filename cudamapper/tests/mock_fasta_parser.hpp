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

#pragma once

#include "gmock/gmock.h"

#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

class MockFastaParser : public io::FastaParser
{
public:
    MOCK_METHOD(number_of_reads_t, get_num_seqences, (), (const, override));
    MOCK_METHOD(const io::FastaSequence&, get_sequence_by_id, (read_id_t sequence_id), (const, override));
};

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
