

#include <vector>
#include <claraparabricks/genomeworks/utils/genomeutils.hpp>

#include "gtest/gtest.h"

namespace claraparabricks
{

namespace genomeworks
{

namespace genomeutils
{

TEST(GenomeUtilsTest, ReverseComplement)
{
    std::string genome("ATCGAACGTATG");
    std::vector<char> complement(genome.size() + 1);
    complement[genome.length()] = '\0';
    reverse_complement(genome.c_str(), genome.length(), complement.data());
    ASSERT_STREQ(complement.data(), "CATACGTTCGAT");
}

} // namespace genomeutils

} // namespace genomeworks

} // namespace claraparabricks
