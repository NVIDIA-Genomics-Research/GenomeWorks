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

#include "../src/myers_gpu.cuh"
#include "../src/needleman_wunsch_cpu.hpp"
#include "cudaaligner_test_cases.hpp"

#include <gtest/gtest.h>

namespace claraparabricks
{

namespace genomeworks
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

} // namespace genomeworks

} // namespace claraparabricks
