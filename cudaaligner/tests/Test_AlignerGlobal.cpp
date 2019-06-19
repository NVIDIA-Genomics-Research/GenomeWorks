/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <random>
#include "gtest/gtest.h"
#include "../src/aligner_global.hpp"
#include "cudaaligner/alignment.hpp"
#include <utils/signed_integer_utils.hpp>
#include <utils/genomeutils.hpp>

namespace genomeworks
{

namespace cudaaligner
{

// Common data structures and functions.
typedef struct
{
    std::vector<std::pair<std::string, std::string>> inputs;
    std::vector<std::string> cigars;
} AlignerTestData;

// Test adding alignments to Aligner objects
TEST(TestCudaAligner, TestAlignmentAddition)
{
    std::unique_ptr<AlignerGlobal> aligner = std::make_unique<AlignerGlobal>(5, 5, 5, 0);
    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));
    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));
    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));

    ASSERT_EQ(3, aligner->num_alignments());

    ASSERT_EQ(StatusType::exceeded_max_length, aligner->add_alignment("ATCGAT", 6, "TACG", 4));
    ASSERT_EQ(StatusType::exceeded_max_length, aligner->add_alignment("ATCG", 4, "TACGAG", 6));

    ASSERT_EQ(3, aligner->num_alignments());

    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));
    ASSERT_EQ(StatusType::success, aligner->add_alignment("ATCG", 4, "TACG", 4));

    ASSERT_EQ(5, aligner->num_alignments());

    ASSERT_EQ(StatusType::exceeded_max_alignments, aligner->add_alignment("ATCG", 4, "TACG", 4));

    ASSERT_EQ(5, aligner->num_alignments());
}

// Test correctness of genome alignment.
std::vector<AlignerTestData> create_aligner_test_cases()
{
    std::vector<AlignerTestData> test_cases;
    AlignerTestData data;

    // Test case 1
    data.inputs = {{"AAAA", "TTAT"}};
    data.cigars = {"4M"};
    test_cases.push_back(data);

    // Test case 2
    data.inputs = {{"ATAAAAAAAA", "AAAAAAAAA"}};
    data.cigars = {"1M1I8M"};
    test_cases.push_back(data);

    // Test case 3
    data.inputs = {{"AAAAAAAAA", "ATAAAAAAAA"}};
    data.cigars = {"1M1D8M"};
    test_cases.push_back(data);

    // Test case 3
    data.inputs = {{"ACTGA", "GCTAG"}};
    data.cigars = {"3M1I1M1D"};
    test_cases.push_back(data);

    // Test case 4
    data.inputs = {{"ACTGA", "GCTAG"}, {"ACTG", "ACTG"}, {"A", "T"}};
    data.cigars = {"3M1I1M1D", "4M", "1M"};
    test_cases.push_back(data);

    return test_cases;
};

class TestAlignerGlobalImpl : public ::testing::TestWithParam<AlignerTestData>
{
public:
    virtual void SetUp()
    {
        param                   = GetParam();
        int64_t max_string_size = 0;
        for (auto& pair : param.inputs)
        {
            max_string_size = std::max(max_string_size, get_size(pair.first));
            max_string_size = std::max(max_string_size, get_size(pair.second));
        }
        max_string_size++;
        aligner = std::make_unique<AlignerGlobal>(max_string_size,
                                                  max_string_size,
                                                  param.inputs.size(),
                                                  0);
        aligner->set_cuda_stream(0);
    }

protected:
    std::unique_ptr<AlignerGlobal> aligner;
    AlignerTestData param;
};

TEST_P(TestAlignerGlobalImpl, TestAlignmentKernel)
{
    const std::vector<std::pair<std::string, std::string>>& inputs = param.inputs;
    const std::vector<std::string>& cigars                         = param.cigars;

    ASSERT_EQ(inputs.size(), cigars.size()) << "Input data length mismatch";

    for (auto& pair : inputs)
    {
        auto& query  = pair.first;
        auto& target = pair.second;
        ASSERT_EQ(StatusType::success, aligner->add_alignment(query.c_str(), query.length(),
                                                              target.c_str(), target.length()))
            << "Could not add alignment to aligner";
    }

    aligner->align_all();
    aligner->sync_alignments();

    const std::vector<std::shared_ptr<Alignment>>& alignments = aligner->get_alignments();
    ASSERT_EQ(get_size(alignments), get_size(inputs));
    for (int32_t a = 0; a < get_size(alignments); a++)
    {
        auto alignment = alignments[a];
        EXPECT_EQ(StatusType::success, alignment->get_status()) << "Alignment status is not success";
        EXPECT_EQ(AlignmentType::global, alignment->get_alignment_type()) << "Alignment type is not global";
        EXPECT_STREQ(cigars[a].c_str(), alignment->convert_to_cigar().c_str()) << "CIGAR doesn't match for alignment " << a;
    }
}

INSTANTIATE_TEST_SUITE_P(TestCudaAligner, TestAlignerGlobalImpl, ::testing::ValuesIn(create_aligner_test_cases()));

// Test performance of kernel for large genomes
std::vector<AlignerTestData> create_aligner_perf_test_cases()
{
    std::vector<AlignerTestData> test_cases;
    AlignerTestData data;

    // Test case 1
    std::minstd_rand rng(1);
    data.inputs = {{genomeworks::genomeutils::generate_random_genome(1000, rng), genomeworks::genomeutils::generate_random_genome(1000, rng)}};
    test_cases.push_back(data);

    // Test case 2
    data.inputs = {{genomeworks::genomeutils::generate_random_genome(9500, rng), genomeworks::genomeutils::generate_random_genome(9000, rng)},
                   {genomeworks::genomeutils::generate_random_genome(3456, rng), genomeworks::genomeutils::generate_random_genome(3213, rng)},
                   {genomeworks::genomeutils::generate_random_genome(20000, rng), genomeworks::genomeutils::generate_random_genome(20000, rng)},
                   {genomeworks::genomeutils::generate_random_genome(15000, rng), genomeworks::genomeutils::generate_random_genome(14000, rng)}};
    test_cases.push_back(data);

    return test_cases;
};

class TestAlignerGlobalImplPerf : public TestAlignerGlobalImpl
{
};

TEST_P(TestAlignerGlobalImplPerf, TestAlignmentKernelPerf)
{
    const std::vector<std::pair<std::string, std::string>>& inputs = param.inputs;

    for (auto& pair : inputs)
    {
        auto& query  = pair.first;
        auto& target = pair.second;
        ASSERT_EQ(StatusType::success, aligner->add_alignment(query.c_str(), query.length(),
                                                              target.c_str(), target.length()));
    }

    aligner->align_all();
    aligner->sync_alignments();

    const std::vector<std::shared_ptr<Alignment>>& alignments = aligner->get_alignments();
    ASSERT_EQ(alignments.size(), inputs.size());
}

INSTANTIATE_TEST_SUITE_P(TestCudaAligner, TestAlignerGlobalImplPerf, ::testing::ValuesIn(create_aligner_perf_test_cases()));
} // namespace cudaaligner
} // namespace genomeworks
