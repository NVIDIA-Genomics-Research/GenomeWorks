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

#include "gtest/gtest.h"
#include <string>
#include <vector>
#include <claraparabricks/genomeworks/cudamapper/overlapper.hpp>

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

    claraparabricks::genomeworks::gw_string_view_t query_view(query_sequence);
    gw_string_view_t target_view(target_sequence);

    cudamapper::Overlap o;
    o.query_start_position_in_read_  = 1;
    o.query_end_position_in_read_    = 636;
    o.target_start_position_in_read_ = 341;
    o.target_end_position_in_read_   = 976;
    o.relative_strand                = RelativeStrand::Forward;

    details::overlapper::extend_overlap_by_sequence_similarity(o, query_view, target_view, 50, 0.8);

    ASSERT_EQ(o.query_start_position_in_read_, 0);
    ASSERT_EQ(o.target_start_position_in_read_, 340);
    ASSERT_EQ(o.query_end_position_in_read_, 660);
    ASSERT_EQ(o.target_end_position_in_read_, 1000);
}

TEST(TestDropOverlaps, drop_overlaps_by_mask)
{
    Overlap o1;
    o1.query_read_id_ = 1;
    Overlap o2;
    o2.query_read_id_ = 2;
    Overlap o3;
    o3.query_read_id_ = 3;
    Overlap o4;
    o4.query_read_id_ = 4;
    Overlap o5;
    o5.query_read_id_ = 5;

    std::vector<Overlap> overlaps{o1, o2, o3, o4, o5};
    std::vector<bool> mask{true, false, true, true, false};
    details::overlapper::drop_overlaps_by_mask(overlaps, mask);
    ASSERT_EQ(overlaps.size(), 2);
    ASSERT_EQ(overlaps[0].query_read_id_, 2);
    ASSERT_EQ(overlaps[1].query_read_id_, 5);

    std::vector<Overlap> empty_overlaps;
    std::vector<bool> empty_bools;
    details::overlapper::drop_overlaps_by_mask(empty_overlaps, empty_bools);
    ASSERT_EQ(empty_overlaps.size(), 0);
}

} // namespace cudamapper

} // namespace genomeworks
} // namespace claraparabricks
