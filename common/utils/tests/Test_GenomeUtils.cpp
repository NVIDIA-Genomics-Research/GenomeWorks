/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <memory>
#include <claragenomics/utils/genomeutils.hpp>

#include "gtest/gtest.h"

namespace claragenomics
{

namespace genomeutils
{

TEST(GenomeUtilsTest, ReverseComplement)
{
    std::string genome("ATCGAACGTATG");
    std::unique_ptr<char[]> complement(new char[genome.length() + 1]);
    complement.get()[genome.length()] = '\0';
    reverse_complement(genome.c_str(), genome.length(), complement.get());
    ASSERT_STREQ(complement.get(), "CATACGTTCGAT");
}

} // genomeutils

} // namespace claragenomics
