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

#include "../src/alignment_impl.hpp"

#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

#include "gtest/gtest.h"
#include <memory>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

using ::testing::ValuesIn;

// Individual tests
TEST(TestAlignmentImplIndividual, Status)
{
    std::unique_ptr<AlignmentImpl> alignment_ = std::make_unique<AlignmentImpl>("A", 1, "T", 1);
    ASSERT_EQ(StatusType::uninitialized, alignment_->get_status()) << "Initial status incorrect";

    alignment_->set_status(StatusType::success);
    ASSERT_EQ(StatusType::success, alignment_->get_status()) << "Status not set properly";
}

TEST(TestAlignmentImplIndividual, Type)
{
    std::unique_ptr<AlignmentImpl> alignment_ = std::make_unique<AlignmentImpl>("A", 1, "T", 1);
    ASSERT_EQ(AlignmentType::unset, alignment_->get_alignment_type()) << "Initial type incorrect";

    alignment_->set_alignment_type(AlignmentType::global_alignment);
    ASSERT_EQ(AlignmentType::global_alignment, alignment_->get_alignment_type()) << "Type not set properly";
}

// Parametrized tests
typedef struct AlignmentTestData
{
    std::string query;
    std::string target;
    std::vector<AlignmentState> alignment;
    bool is_optimal;
    FormattedAlignment formatted_alignment;
    std::string cigar_basic;
    std::string cigar_extended;
} AlignmentTestData;

std::vector<AlignmentTestData> create_alignment_test_cases()
{
    std::vector<AlignmentTestData> test_cases;
    AlignmentTestData data;

    // Test case 1
    data.query     = "AAAA";
    data.target    = "TTATG";
    data.alignment = {
        AlignmentState::mismatch,
        AlignmentState::mismatch,
        AlignmentState::match,
        AlignmentState::mismatch,
        AlignmentState::insertion};
    data.is_optimal          = true;
    data.formatted_alignment = FormattedAlignment{"AAAA-", "xx|x ", "TTATG"};
    data.cigar_basic         = "4M1I";
    data.cigar_extended      = "2X1=1X1I";
    test_cases.push_back(data);

    // Test case 2
    data.query     = "CGATAATG";
    data.target    = "CATAA";
    data.alignment = {
        AlignmentState::deletion,
        AlignmentState::mismatch,
        AlignmentState::match,
        AlignmentState::match,
        AlignmentState::match,
        AlignmentState::match,
        AlignmentState::deletion,
        AlignmentState::deletion};
    data.is_optimal          = true;
    data.formatted_alignment = FormattedAlignment{"CGATAATG", " x||||  ", "-CATAA--"};
    data.cigar_basic         = "1D5M2D";
    data.cigar_extended      = "1D1X4=2D";
    test_cases.push_back(data);

    // Test case 3
    data.query     = "GTTAG";
    data.target    = "AAGTCTAGAA";
    data.alignment = {
        AlignmentState::insertion,
        AlignmentState::insertion,
        AlignmentState::match,
        AlignmentState::match,
        AlignmentState::insertion,
        AlignmentState::match,
        AlignmentState::match,
        AlignmentState::match,
        AlignmentState::insertion,
        AlignmentState::insertion,
    };
    data.is_optimal          = true;
    data.formatted_alignment = FormattedAlignment{"--GT-TAG--", "  || |||  ", "AAGTCTAGAA"};
    data.cigar_basic         = "2I2M1I3M2I";
    data.cigar_extended      = "2I2=1I3=2I";
    test_cases.push_back(data);

    // Test case 4
    data.query     = "GTTACA";
    data.target    = "GATTCA";
    data.alignment = {
        AlignmentState::match,
        AlignmentState::insertion,
        AlignmentState::match,
        AlignmentState::match,
        AlignmentState::deletion,
        AlignmentState::match,
        AlignmentState::match};
    data.is_optimal          = false; // this example is optimal, but is_optimal = false does only mean it is an upper bound
    data.formatted_alignment = FormattedAlignment{"G-TTACA", "| || ||", "GATT-CA"};
    data.cigar_basic         = "1M1I2M1D2M";
    data.cigar_extended      = "1=1I2=1D2=";
    test_cases.push_back(data);

    return test_cases;
};

class TestAlignmentImpl : public ::testing::TestWithParam<AlignmentTestData>
{
public:
    void SetUp()
    {
        param_     = GetParam();
        alignment_ = std::make_unique<AlignmentImpl>(param_.query.c_str(),
                                                     param_.query.size(),
                                                     param_.target.c_str(),
                                                     param_.target.size());
        alignment_->set_alignment(param_.alignment, param_.is_optimal);
    }

protected:
    std::unique_ptr<AlignmentImpl> alignment_;
    AlignmentTestData param_;
};

TEST_P(TestAlignmentImpl, StringGetters)
{
    ASSERT_EQ(param_.query, alignment_->get_query_sequence()) << "Query doesn't match original string";
    ASSERT_EQ(param_.target, alignment_->get_target_sequence()) << "Target doesn't match original string";
}

TEST_P(TestAlignmentImpl, AlignmentState)
{
    const std::vector<AlignmentState>& al_read = alignment_->get_alignment();
    ASSERT_EQ(param_.alignment.size(), al_read.size());
    for (int64_t i = 0; i < get_size(param_.alignment); i++)
    {
        ASSERT_EQ(param_.alignment[i], al_read[i]);
    }
}

TEST_P(TestAlignmentImpl, AlignmentIsOptimal)
{
    ASSERT_EQ(param_.is_optimal, alignment_->is_optimal());
}

TEST_P(TestAlignmentImpl, AlignmentFormatting)
{
    FormattedAlignment formatted_alignment = alignment_->format_alignment();
    ASSERT_EQ(param_.formatted_alignment.query, formatted_alignment.query);
    ASSERT_EQ(param_.formatted_alignment.pairing, formatted_alignment.pairing);
    ASSERT_EQ(param_.formatted_alignment.target, formatted_alignment.target);
}

TEST_P(TestAlignmentImpl, CigarFormattingBasic)
{
    std::string cigar = alignment_->convert_to_cigar(CigarFormat::basic);
    ASSERT_EQ(param_.cigar_basic, cigar);
}

TEST_P(TestAlignmentImpl, CigarFormattingExtended)
{
    std::string cigar = alignment_->convert_to_cigar(CigarFormat::extended);
    ASSERT_EQ(param_.cigar_extended, cigar);
}

INSTANTIATE_TEST_SUITE_P(TestAlignment, TestAlignmentImpl, ValuesIn(create_alignment_test_cases()));
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
