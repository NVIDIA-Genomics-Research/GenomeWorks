/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "../src/myers_gpu.cuh"
#include "../src/needleman_wunsch_cpu.hpp"
#include "cudaaligner_test_cases.hpp"

#include <gtest/gtest.h>

namespace claragenomics
{

namespace cudaaligner
{

class TestMyersEditDistance : public ::testing::TestWithParam<TestCaseData>
{
};

TEST_P(TestMyersEditDistance, TestCases)
{
    TestCaseData t = GetParam();

    int32_t d         = myers_compute_edit_distance(t.target, t.query);
    matrix<int32_t> r = needleman_wunsch_build_score_matrix_naive(t.target, t.query);
    int32_t reference = r(r.num_rows() - 1, r.num_cols() - 1);
    ASSERT_EQ(d, reference);
}

class TestMyersScoreMatrix : public ::testing::TestWithParam<TestCaseData>
{
};

TEST_P(TestMyersScoreMatrix, TestCases)
{
    TestCaseData t = GetParam();

    matrix<int32_t> m = myers_get_full_score_matrix(t.target, t.query);
    matrix<int32_t> r = needleman_wunsch_build_score_matrix_naive(t.target, t.query);

    ASSERT_EQ(m.num_rows(), r.num_rows());
    ASSERT_EQ(m.num_cols(), r.num_cols());

    for (int32_t j = 0; j < m.num_cols(); ++j)
    {
        for (int32_t i = 0; i < m.num_rows(); ++i)
        {
            EXPECT_EQ(m(i, j), r(i, j)) << "index: (" << i << "," << j << ")";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TestMyersAlgorithm, TestMyersEditDistance, ::testing::ValuesIn(create_cudaaligner_test_cases()));
INSTANTIATE_TEST_SUITE_P(TestMyersAlgorithm, TestMyersScoreMatrix, ::testing::ValuesIn(create_cudaaligner_test_cases()));

} // namespace cudaaligner
} // namespace claragenomics
