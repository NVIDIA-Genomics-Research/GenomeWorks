/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "gmock/gmock.h"

#include <claragenomics/io/fasta_parser.hpp>

namespace claragenomics
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
} // namespace claragenomics
