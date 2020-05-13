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
#include <string_view>
#include <vector>
#include "../src/cudamapper_utils.cpp"

namespace claragenomics
{
namespace cudamapper
{

TEST(KmerizeStringTest, AAACCTTCTCT_k4_s1)
{
    std::string test_str("AAACCTTCTCT");
    // AAAC AACC ACCT CCTT CTTC TTCT TCTC CTCT (n = 8)
    std::vector<std::string> kmers = split_into_kmers(test_str, 4, 1);
    ASSERT_EQ(kmers.size(), 8);
    ASSERT_STREQ(kmers[0].c_str(), "AAAC");
    ASSERT_STREQ(kmers[7].c_str(), "CTCT");
}

TEST(KmerizeStringTest, empty_string)
{
    std::string s("");
    std::vector<std::string> kmers = split_into_kmers(s, 4, 1);
    ASSERT_EQ(kmers.size(), 0);
}

TEST(CountSharedElementsTest, Shared_ints_correctly_counted)
{
    std::vector<int> test_ints_a{1, 2, 5, 10, 1000, 10000};
    std::vector<int> test_ints_b{1, 3, 5, 10, 20000};
    std::size_t shared_count = count_shared_elements(test_ints_a, test_ints_b);
    ASSERT_EQ(shared_count, 3);
}

TEST(CountSharedElementTest, shared_strings_correctly_counted)
{
    std::vector<std::string> test_str_a{"A", "AA", "BET", "CAT"};
    std::vector<std::string> test_str_b{"A", "B", "BEST", "BET", "cat", "CAT", "CHAT"};
    std::sort(test_str_a.begin(), test_str_a.end());
    std::sort(test_str_b.begin(), test_str_b.end());
    std::size_t shared_count = count_shared_elements(test_str_a, test_str_b);
    ASSERT_EQ(shared_count, 3);
}

TEST(CountSharedElementsTest, string_views_can_be_compared)
{
    std::string_view s("first");
    std::string_view t("first");
    std::string_view w("second");
    std::vector<std::string_view> test_a{s, w};
    std::vector<std::string_view> test_b{t};
    std::size_t shared_count = count_shared_elements(test_a, test_b);
    ASSERT_EQ(shared_count, 1);
}

TEST(CountSharedElementsTest, empty_vectors_counted_correctly)
{
    std::vector<int> first;
    std::vector<int> second;
    std::vector<int> third{1};
    std::size_t empty_shared_count = count_shared_elements(first, second);
    std::size_t no_overlap_count   = count_shared_elements(first, third);
    ASSERT_EQ(empty_shared_count, 0);
    ASSERT_EQ(no_overlap_count, 0);
}

TEST(SimilarityTest, similarity_of_identical_seqs_is_1)
{
    std::string a("AAACCTATGAGGG");
    std::string b("AAACCTATGAGGG");
    std::string long_b("AAACCTATGAGGGAAACCTATGAGGG");
    float sim              = sequence_jaccard_similarity(a, b, 4, 1);
    float long_containment = sequence_jaccard_containment(a, long_b, 4, 1);
    ASSERT_EQ(sim, 1.0);
    ASSERT_EQ(long_containment, 1.0);
}
TEST(SimilarityTest, similarity_of_disjoint_seqs_is_0)
{
    std::string a("AAACCTATGAGGG");
    std::string b("CCCAATTTAAATT");
    float sim = sequence_jaccard_similarity(a, b, 4, 1);
    ASSERT_EQ(sim, 0.0);
}
TEST(SimilarityTest, similarity_of_similar_seqs_is_accurate_estimate)
{
    std::string a("AAACCTATGAGGG");
    std::string b("AAACCTAAGAGGG");
    float sim = sequence_jaccard_similarity(a, b, 4, 1);
    ASSERT_GT(sim, 0.0);
    ASSERT_LT(sim, 1.0);
}

} // namespace cudamapper
} // namespace claragenomics