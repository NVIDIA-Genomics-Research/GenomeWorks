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

#include "../src/aligner_global_myers_banded.hpp"
#include "cudaaligner_file_location.hpp"
#include <claraparabricks/genomeworks/cudaaligner/alignment.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <fstream>

namespace
{

constexpr int32_t word_size = 32;

struct TestCase
{
    std::string query;
    std::string target;
    int32_t edit_distance;
};

std::vector<TestCase> create_band_test_cases()
{
    std::vector<TestCase> data;

    std::unique_ptr<claraparabricks::genomeworks::io::FastaParser> target_parser = claraparabricks::genomeworks::io::create_kseq_fasta_parser(std::string(CUDAALIGNER_BENCHMARK_DATA_DIR) + "/target_AlignerGlobal.fasta", 0, false);
    std::unique_ptr<claraparabricks::genomeworks::io::FastaParser> query_parser  = claraparabricks::genomeworks::io::create_kseq_fasta_parser(std::string(CUDAALIGNER_BENCHMARK_DATA_DIR) + "/query_AlignerGlobal.fasta", 0, false);

    std::ifstream edit_dist_file(std::string(CUDAALIGNER_BENCHMARK_DATA_DIR) + "/result_ApproximateBandedMyers.txt");
    std::string test_case;
    int32_t edit_distance;

    assert(target_parser->get_num_seqences() == query_parser->get_num_seqences());
    claraparabricks::genomeworks::read_id_t read = 0;
    while (edit_dist_file >> test_case >> edit_distance)
    {
        assert(target_parser->get_sequence_by_id(read).name == test_case);
        assert(query_parser->get_sequence_by_id(read).name == test_case);
        data.push_back({query_parser->get_sequence_by_id(read).seq, target_parser->get_sequence_by_id(read).seq, edit_distance});
    }

    return data;
}

class TestApproximateBandedMyers : public ::testing::TestWithParam<TestCase>
{
};

void implicit_new_entries_test_impl(const std::string& query, const std::string& target, const std::string& expected_cigar)
{
    using namespace claraparabricks::genomeworks::cudaaligner;
    using namespace claraparabricks::genomeworks;
    const int32_t max_bw             = 7;
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    std::unique_ptr<Aligner> aligner = std::make_unique<AlignerGlobalMyersBanded>(-1,
                                                                                  max_bw,
                                                                                  allocator,
                                                                                  nullptr,
                                                                                  0);
    ASSERT_EQ(StatusType::success, aligner->add_alignment(query.c_str(), query.length(), target.c_str(), target.length()))
        << "Could not add alignment to aligner";
    aligner->align_all();
    aligner->sync_alignments();
    const std::vector<std::shared_ptr<Alignment>>& alignments = aligner->get_alignments();
    ASSERT_EQ(get_size(alignments), 1);
    ASSERT_EQ(alignments[0]->get_status(), StatusType::success);
    ASSERT_EQ(alignments[0]->is_optimal(), false);
    ASSERT_EQ(alignments[0]->convert_to_cigar(), expected_cigar);
}

} // namespace

TEST(TestApproximateBandedMyersStatic, ImplicitNWEntries1)
{
    // The banded Myers implementation uses implicit entries for the 0-th row of the NW matrix (and the band),
    // which is assumed to be the worst case value. This does not pose a problem on the top left block and
    // the diagonal band, but on the lower right block there are cases where the backtrace runs through this worst case
    // - which is technically still part of the band. This tests this specific corner case.

    // * = tested implicit 0-row entry
    // Band of the NW matrix:
    // top left block of NW m.|  diagonal band (each column shifted by one row)                                                                 | bottom right block
    //   NW_(i,j), index i of first shown row:
    //    1  1  1  1  1  1  1 |   2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19    20 |  20 20 20 20 20 20
    //
    //       A  A  C  C  G  G |   T     T     A     A     A     A     C     C     C     C     G     G     G     G     G     T     T     A     A |   A  C  G  G  T  T
    // row 0 (implicit):      |                                                                                                               * |   *
    // A  1  0  1  2  3  4  5 |A  5  C  5  C  5  G  5  G  5  T  5  T  5  A  5  A  5  C  5  C  5  G  5  G  5  T  6  T  7  A  8  A  9  C 10  C 11 |C 12 12 13 14 15 16
    // A  2  1  0  1  2  3  4 |C  4  C  4  G  4  G  4  T  4  T  4  A  4  A  4  C  4  C  4  G  4  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 11 |G 12 13 12 13 14 15
    // C  3  2  1  0  1  2  3 |C  3  G  3  G  3  T  3  T  3  A  3  A  3  C  3  C  3  G  4  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 11  G 12 |G 12 13 13 12 13 14
    // C  4  3  2  1  0  1  2 |G  2  G  2  T  2  T  2  A  2  A  2  C  2  C  2  G  3  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 11  G 12  T 13 |T 13 13 14 13 12 13
    // G  5  4  3  2  1  0  1 |G  1  T  1  T  1  A  1  A  1  C  2  C  2  G  3  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 11  G 12  T 12  T 12 |T 13 14 14 14 13 12
    // G  6  5  4  3  2  1  0 |T  0  T  0  A  0  A  0  C  1  C  2  G  3  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 10  G 11  T 11  T 11  T 12 |T 13 14 15 15 14 13
    // T  7  6  5  4  3  2  1 |T  1  A  1  A  1  C  1  C  2  G  3  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 10  G 10  T 10  T 10  T 11  T 12 |T 13 14 15 16 15 14

    implicit_new_entries_test_impl("AACCGGTTAACCGGTTAACCGGTTTT",
                                   "AACCGGTTAAAACCCCGGGGGTTAAACGGTT",
                                   "10M2I2M2I7M3I5M2D");
}

TEST(TestApproximateBandedMyersStatic, ImplicitNWEntries2)
{
    // * = tested implicit 0-row entry
    // Band of the NW matrix:
    // top left block of NW mat, |  diagonal band (each column shifted by one row)                                                           | bottom right block
    //   NW_(i,j), index i of first shown row:
    //    1  1  1  1  1  1  1  1 |   2     3     4     5     6     7     8     9    10    11    12    13    14    15    16    17    18    19 |   19 19 19 19 19 19
    //
    //       A  A  C  C  G  G  T |   T     A     A     A     A     C     C     C     C     G     G     G     G     G     T     T     A     A |    C  C  G  G  T  T
    // row 0 (implicit):         |                                                                                                         * |
    // A  1  0  1  2  3  4  5  6 |A  6  C  6  C  6  G  6  G  6  T  6  T  6  A  6  A  6  C  6  C  6  G  6  G  6  T  7  T  7  A  8  A  8  C  9 |C   9 10 11 12 13 14
    // A  2  1  0  1  2  3  4  5 |C  5  C  5  G  5  G  5  T  5  T  5  A  5  A  5  C  5  C  5  G  5  G  5  T  6  T  7  A  8  A  9  C  9  C 10 |C   9  9 10 11 12 13
    // C  3  2  1  0  1  2  3  4 |C  4  G  4  G  4  T  4  T  4  A  4  A  4  C  4  C  4  G  4  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 11 |G  10 10  9 10 11 12
    // C  4  3  2  1  0  1  2  3 |G  3  G  3  T  3  T  3  A  3  A  3  C  3  C  3  G  4  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 11  G 12 |G  11 11 10  9 10 11
    // G  5  4  3  2  1  0  1  2 |G  2  T  2  T  2  A  2  A  2  C  2  C  2  G  3  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 11  G 12  T 13 |T  12 12 11 10  9 10
    // G  6  5  4  3  2  1  0  1 |T  1  T  1  A  1  A  1  C  2  C  2  G  3  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 11  G 12  T 12  T 13 |T  13 13 12 11 10  9
    // T  7  6  5  4  3  2  1  0 |T  0  A  0  A  0  C  1  C  2  G  3  G  4  T  5  T  6  A  7  A  8  C  9  C 10  G 10  G 11  T 11  T 12  T 13 |T  14 14 13 12 11 10
    implicit_new_entries_test_impl("AACCGGTTAACCGGTTAACCGGTTT",
                                   "AACCGGTTAAAACCCCGGGGGTTAACCGGTT",
                                   "10M2I2M2I3M2I3M1I6M1D");
}

TEST_P(TestApproximateBandedMyers, EditDistanceMonotonicallyDecreasesWithBandWidth)
{
    using namespace claraparabricks::genomeworks::cudaaligner;
    using namespace claraparabricks::genomeworks;

    TestCase t = GetParam();

    DefaultDeviceAllocator allocator = create_default_device_allocator();

    int32_t last_edit_distance      = std::numeric_limits<int32_t>::max();
    int32_t last_bw                 = -1;
    std::vector<int32_t> bandwidths = {2, 4, 16, 31, 32, 34, 63, 64, 66, 255, 256, 258, 1023, 1024, 1026, 2048};
    for (const int32_t max_bw : bandwidths)
    {
        if (max_bw % word_size == 1)
            continue; // not supported
        std::unique_ptr<Aligner> aligner = std::make_unique<AlignerGlobalMyersBanded>(-1,
                                                                                      max_bw,
                                                                                      allocator,
                                                                                      nullptr,
                                                                                      0);

        ASSERT_EQ(StatusType::success, aligner->add_alignment(t.query.c_str(), t.query.length(),
                                                              t.target.c_str(), t.target.length()))
            << "Could not add alignment to aligner";
        aligner->align_all();
        aligner->sync_alignments();
        const std::vector<std::shared_ptr<Alignment>>& alignments = aligner->get_alignments();
        ASSERT_EQ(get_size(alignments), 1);
        if (alignments[0]->get_status() == StatusType::success)
        {
            const int32_t edit_distance = alignments[0]->get_edit_distance();
            ASSERT_LE(edit_distance, last_edit_distance) << "for max bandwidth = " << max_bw << " vs. max bandwidth = " << last_bw;

            if (edit_distance > t.edit_distance)
            {
                ASSERT_EQ(alignments[0]->is_optimal(), false) << "for max bandwidth = " << max_bw << " the alignment should be approximate.";
            }

            if (max_bw == bandwidths.back())
            {
                ASSERT_EQ(alignments[0]->is_optimal(), true) << "for max bandwidth = " << max_bw << " the alignment should be optimal.";
            }

            last_edit_distance = edit_distance;
            last_bw            = max_bw;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TestApproximateBandedMyersInstance, TestApproximateBandedMyers, ::testing::ValuesIn(create_band_test_cases()));
