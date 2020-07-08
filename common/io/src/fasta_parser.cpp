

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
