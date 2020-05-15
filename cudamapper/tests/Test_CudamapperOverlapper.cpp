/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"
#include <string>
#include <vector>
#include "../include/claragenomics/cudamapper/overlapper.hpp"

namespace claraparabricks
{
namespace genomeworks
{

namespace cudamapper
{

TEST(TestOverlapExtension, short_forward_head_overlap_properly_extended)
{
    std::string query_sequence("ACCGCCACCAATATCCATGTGACC"
                               "TCGCACGGTACGGAATTTACCCTACAAACCCCAACCGGTAGCGTCGATGTTCTGCTGCCGTTGCCGGGGCGTCACAATATTGCGAATGCGCTGGCA"
                               "GCCGCTGCGCTCTCCATGTCCGTGGGCGCAACGCTTGATGCTATCAAAGCGGGGCTGGCA"
                               "AATCTGAAAGCTGTTCCAGGCCGTCTGTTCCCCATCCAACTGGCAGAAAACCAGTTGCTG"
                               "CTCGACGACTCCTACAACGCCAATGTCGGTTCAATGACTGCAGCAGTCCAGGTACTGGCT"
                               "GAAATGCCGGGCTACCGCGTGCTGGTGGTGGGCGATATGGCGGAACTGGGCGCTGAAAGC"
                               "GAAGCCTGCCATGTACAGGTGGGCGAGGCGGCAAAAGCTGCTGGTATTGACCGCGTGTTA"
                               "AGCGTGGGTAAACAAAGCCATGCTATCAGCACCGCCAGCGGCGTTGGCGAACATTTTGCT"
                               "GATAAAACTGCGTTAATTACGCGTCTTAAATTACTGATTGCTGAGCAACAGGTAATTACG"
                               "ATTTTAGTTAAGGGTTCACGTAGTGCCGCCATGGAAGAGGTAGTACGCGCTTTACAGGAG"
                               "AATGGGACATGTTAGTTTGGCTGGCCGAACATTTGGTCAAATATTATTCCGGCTTTAACG"
                               "TCTTTTCCTATCTGACGTTTCGCGCCATCGTCAGCCTGCTGACCGCGCTGTTCATCTCAT"
                               "TGTGGATGGGCCCGCGTATGATTGCTCATTTGCAAAAACTTTCCTTTGGTCAGGTGGTGC"
                               "GTAACGACGGTCCTGAATCACACTTCAGCAAGCGCGGTACGCCGACCATGGGCGGGATTA"
                               "TGATCCTGACGGCGATTGTGATCTCCGTACTGCTGTGGGCTTACCCGTCCAATCCGTACG"
                               "TCTGGTGCGTGTTGGTGGTGCTGGTAGGTTACGGTGTTATTGGCTTTGTTGATGATTATC"
                               "GCAAAGTGGTGCGTAAAGACACCAAAGGGTTGATCGCTCG");
    std::string target_sequence("CAACAACGACATCGGTGTACCGA"
                                "TGACGCTGTTGCGCTTAACGCCGGAATACGATTACGC"
                                "AGTTATTGAACTTGGCGCGAACCATCAGGGCGAAATAGCCTGGACTGTGAGTCTGACTCG"
                                "CCCGGAAGCTGCGCTGGTCAACAACCTGGCAGCGGCGCATCTGGAAGGTTTTGGCTCGCT"
                                "TGCGGGTGTCGCGAAAGCGAAAGGTGAAATCTTTAGCGGCCTGCCGGAAAACGGTATCGC"
                                "CATTATGAACGCCGACAACAACGACTGGCTGAACTGGCAGAGCGTAATTGGCTCACGCAA"
                                "AGTGTGGCGTTTCTCACCCAATGCCGCCAACAGCGATTTCACCGCCACCAATATCCATGT"
                                "GACCTCGCACGGTACGGAATTTACCCTACAAACCCCAACCGGTAGCGTCGATGTTCTGCT"
                                "GCCGTTGCCGGGGCGTCACAATATTGCGAATGCGCTGGCAGCCGCTGCGCTCTCCATGTC"
                                "CGTGGGCGCAACGCTTGATGCTATCAAAGCGGGGCTGGCAAATCTGAAAGCTGTTCCAGG"
                                "CCGTCTGTTCCCCATCCAACTGGCAGAAAACCAGTTGCTGCTCGACGACTCCTACAACGC"
                                "CAATGTCGGTTCAATGACTGCAGCAGTCCAGGTACTGGCTGAAATGCCGGGCTACCGCGT"
                                "GCTGGTGGTGGGCGATATGGCGGAACTGGGCGCTGAAAGCGAAGCCTGCCATGTACAGGT"
                                "GGGCGAGGCGGCAAAAGCTGCTGGTATTGACCGCGTGTTAAGCGTGGGTAAACAAAGCCA"
                                "TGCTATCAGCACCGCCAGCGGCGTTGGCGAACATTTTGCTGATAAAACTGCGTTAATTAC"
                                "GCGTCTTAAATTACTGATTGCTGAGCAACAGGTAATTACGATTTTAGTTAAGGGTTCACG"
                                "TAGTGCCGCCATGGAAGAGGTAGTACGCGCTTTACAGGAGAATGGGACATGTTAGTTTGG"
                                "CTGGCCGAACATTTGGTCAAATATTATTCCGGCTTTAACG");

    claraparabricks::genomeworks::cga_string_view_t query_view(query_sequence);
    cga_string_view_t target_view(target_sequence);

    cudamapper::Overlap o;
    o.query_start_position_in_read_  = 1;
    o.query_end_position_in_read_    = 636;
    o.target_start_position_in_read_ = 341;
    o.target_end_position_in_read_   = 976;
    o.relative_strand                = RelativeStrand::Forward;

    Overlapper::extend_overlap_by_sequence_similarity(o, query_view, target_view, 50, 0.8);

    ASSERT_EQ(o.query_start_position_in_read_, 0);
    ASSERT_EQ(o.target_start_position_in_read_, 340);
    ASSERT_EQ(o.query_end_position_in_read_, 660);
    ASSERT_EQ(o.target_end_position_in_read_, 1000);
}

} // namespace cudamapper

} // namespace genomeworks
} // namespace claraparabricks