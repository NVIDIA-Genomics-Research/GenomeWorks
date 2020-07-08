

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
