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
#include "../src/cudamapper_utils.cpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

TEST(KmerizeStringTest, AAACCTTCTCT_k4_s1)
{
    std::string test_str("AAACCTTCTCT");
    gw_string_view_t test_view(test_str);
    // AAAC AACC ACCT CCTT CTTC TTCT TCTC CTCT (n = 8)
    std::vector<gw_string_view_t> kmers = split_into_kmers(test_view, 4, 1);
    ASSERT_EQ(kmers.size(), 8);
    ASSERT_EQ(kmers[0].compare(gw_string_view_t("AAAC")), 0);
    ASSERT_EQ(kmers[7].compare(gw_string_view_t("CTCT")), 0);
}

TEST(KmerizeStringTest, empty_string)
{
    std::string s("");
    std::vector<gw_string_view_t> kmers = split_into_kmers(s, 4, 1);
    ASSERT_EQ(kmers.size(), 1);
    ASSERT_EQ(kmers[0].compare(""), 0);
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
    float sim = sequence_jaccard_similarity(a, b, 4, 1);
    ASSERT_EQ(sim, 1.0);
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

} // namespace genomeworks
} // namespace claraparabricks