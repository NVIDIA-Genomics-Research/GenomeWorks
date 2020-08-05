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

#include "../benchmarks/multi_batch.hpp"
#include <claraparabricks/genomeworks/cudapoa/utils.hpp>
#include "file_location.hpp"

#include "gtest/gtest.h"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

typedef struct
{
    std::string data_file;
    std::string golden_file;
    int32_t batches;
} End2EndBatchTestParam;

std::vector<End2EndBatchTestParam> getCudapoaBatchEnd2EndTestCases()
{
    std::vector<End2EndBatchTestParam> test_cases;

    End2EndBatchTestParam test1{};
    test1.data_file   = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt";
    test1.golden_file = std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-golden-value.txt";

    test1.batches = 2;
    test_cases.push_back(test1);

    test1.batches = 4;
    test_cases.push_back(test1);
    return test_cases;
}

// Testing correctness of batched POA consensus generation when run over a large number
// of POAs.
class TestCudapoaBatchEnd2End : public ::testing::TestWithParam<End2EndBatchTestParam>
{
public:
    void SetUp()
    {
        param_       = GetParam();
        multi_batch_ = std::make_unique<MultiBatch>(param_.batches, param_.data_file);
    }

    void TearDown()
    {
        multi_batch_.reset();
    }

    void run_test()
    {
        multi_batch_->process_batches();
        std::string genome = multi_batch_->assembly();

        std::string golden_genome = parse_golden_value_file(param_.golden_file);

        ASSERT_EQ(golden_genome, genome);
    }

private:
    std::unique_ptr<MultiBatch> multi_batch_;
    End2EndBatchTestParam param_;
};

TEST_P(TestCudapoaBatchEnd2End, TestCorrectness)
{
    run_test();
}

INSTANTIATE_TEST_SUITE_P(TestEnd2End, TestCudapoaBatchEnd2End, ::testing::ValuesIn(getCudapoaBatchEnd2EndTestCases()));

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
