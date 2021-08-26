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
#include "cudaaligner_file_location.hpp"

#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

#include "gtest/gtest.h"
#include <memory>
#include <fstream>

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

AlignmentState get_alignment_state(const std::string s)
{
    if (s == "match")
        return AlignmentState::match;
    if (s == "mismatch")
        return AlignmentState::mismatch;
    if (s == "insertion")
        return AlignmentState::insertion;
    if (s == "deletion")
        return AlignmentState::deletion;
    assert(false);
    return AlignmentState(0);
}

std::vector<AlignmentTestData> create_alignment_test_cases()
{
    std::vector<AlignmentTestData> test_cases;
    AlignmentTestData data;

    std::unique_ptr<claraparabricks::genomeworks::io::FastaParser> target_parser = claraparabricks::genomeworks::io::create_kseq_fasta_parser(std::string(CUDAALIGNER_BENCHMARK_DATA_DIR) + "/target_AlignmentImpl.fasta", 0, false);
    std::unique_ptr<claraparabricks::genomeworks::io::FastaParser> query_parser  = claraparabricks::genomeworks::io::create_kseq_fasta_parser(std::string(CUDAALIGNER_BENCHMARK_DATA_DIR) + "/query_AlignmentImpl.fasta", 0, false);
    std::ifstream result_file(std::string(CUDAALIGNER_BENCHMARK_DATA_DIR) + "/result_AlignmentImpl.txt");
    std::string result_line;
    read_id_t read_id = 0;

    while (getline(result_file, result_line))
    {
        data = {};
        std::stringstream linestream(result_line);
        std::string test_case, formatted_field_1, formatted_field_2, formatted_field_3, optimal, alignment_field, alignment_state;
        while (getline(linestream, test_case, ';'))
        {
            assert(query_parser->get_sequence_by_id(read_id).name == test_case);
            assert(target_parser->get_sequence_by_id(read_id).name == test_case);
            data.query  = query_parser->get_sequence_by_id(read_id).seq;
            data.target = target_parser->get_sequence_by_id(read_id).seq;
            getline(linestream, formatted_field_1, ';');
            getline(linestream, formatted_field_2, ';');
            getline(linestream, formatted_field_3, ';');
            getline(linestream, data.cigar_basic, ';');
            getline(linestream, data.cigar_extended, ';');
            getline(linestream, optimal, ';');
            getline(linestream, alignment_field, ';');

            data.formatted_alignment = FormattedAlignment{formatted_field_1, formatted_field_2, formatted_field_3};
            data.is_optimal          = ((optimal == "true") ? true : false);
            std::istringstream alignstream(alignment_field);
            while (getline(alignstream, alignment_field, '/'))
            {
                data.alignment.push_back(get_alignment_state(alignment_field));
            }
        }
        read_id++;
        test_cases.push_back(data);
    }

    return test_cases;
}

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
