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
#include <claraparabricks/genomeworks/cudaaligner/alignment.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

#include <gtest/gtest.h>
#include <algorithm>
#include <limits>

namespace
{

constexpr int32_t word_size = 32;
int32_t get_max_sequence_length(std::vector<std::pair<std::string, std::string>> const& inputs)
{
    using claraparabricks::genomeworks::get_size;
    int64_t max_string_size = 0;
    for (auto const& pair : inputs)
    {
        max_string_size = std::max(max_string_size, get_size(pair.first));
        max_string_size = std::max(max_string_size, get_size(pair.second));
    }
    return static_cast<int32_t>(max_string_size);
}

struct TestCase
{
    std::string query;
    std::string target;
    int32_t edit_distance;
};

std::vector<TestCase> create_band_test_cases()
{
    std::vector<TestCase> data;
    data.push_back({"AGGGCGAATATCGCCTCCCGCATTAAGCTGTACCTTCCAGCCCCGCCGGTAATTCCAGCCGGTTGAAGCCACGTCTGCCACGGCACAATGTTTTCGCTTTGCCCGGTGACGGATTTAATCCACCACAG", "AGGGCGAATATCGCCTCCGCATTAAACTGTACTTCCCAGCCCCGCCAGTATTCCAGCGGGTTGAAGCCGCGTCTGCCACAGCGCAATGTTTTCTTTGCCCACGGTGACCGGTTTAGTCACTACAGTTGC", 23});
    return data;
}

class TestApproximateBandedMyers : public ::testing::TestWithParam<TestCase>
{
};

} // namespace

TEST_P(TestApproximateBandedMyers, EditDistanceMonotonicallyDecreasesWithBandWidth)
{
    using namespace claraparabricks::genomeworks::cudaaligner;
    using namespace claraparabricks::genomeworks;

    TestCase t = GetParam();

    DefaultDeviceAllocator allocator = create_default_device_allocator();

    const int32_t max_string_size = std::max(get_size<int32_t>(t.query), get_size<int32_t>(t.target));

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
};

INSTANTIATE_TEST_SUITE_P(TestApproximateBandedMyersInstance, TestApproximateBandedMyers, ::testing::ValuesIn(create_band_test_cases()));
