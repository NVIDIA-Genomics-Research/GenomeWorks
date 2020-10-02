/*
* Copyright 2020 NVIDIA CORPORATION.
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

#include <cudaextender_file_location.hpp>
#include <claraparabricks/genomeworks/cudaextender/utils.hpp>
#include <claraparabricks/genomeworks/cudaextender/extender.hpp>
#include <claraparabricks/genomeworks/utils/pinned_host_vector.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

struct End2EndTestParam
{
    std::string query_file;
    std::string target_file;
    std::string seed_pairs_file;
    std::string golden_scored_segment_pairs_file;
    int32_t score_threshold;
};

std::vector<End2EndTestParam> getCudaextenderEnd2EndTestCases()
{
    std::vector<End2EndTestParam> test_cases;

    End2EndTestParam test1{};
    test1.query_file                       = std::string(CUDAEXTENDER_DATA_DIR) + "/sample.fa";
    test1.target_file                      = std::string(CUDAEXTENDER_DATA_DIR) + "/sample.fa";
    test1.seed_pairs_file                  = std::string(CUDAEXTENDER_DATA_DIR) + "/sample_seed_pairs.csv";
    test1.golden_scored_segment_pairs_file = std::string(CUDAEXTENDER_DATA_DIR) + "/sample_scored_segment_pairs.csv";
    test1.score_threshold                  = 3000;
    test_cases.push_back(test1);

    return test_cases;
}

// Testing correctness of UngappedXDrop extension End2End with default scoring matrix
class TestCudaextenderEnd2End : public ::testing::TestWithParam<End2EndTestParam>
{
public:
    void SetUp()
    {
        param_ = GetParam();
        // Define Scoring Matrix
        int32_t score_matrix[NUC2]       = {91, -114, -31, -123, -1000, -1000, -100, -9100,
                                      -114, 100, -125, -31, -1000, -1000, -100, -9100,
                                      -31, -125, 100, -114, -1000, -1000, -100, -9100,
                                      -123, -31, -114, 91, -1000, -1000, -100, -9100,
                                      -1000, -1000, -1000, -1000, -1000, -1000, -1000, -9100,
                                      -1000, -1000, -1000, -1000, -1000, -1000, -1000, -9100,
                                      -100, -100, -100, -100, -1000, -1000, -100, -9100,
                                      -9100, -9100, -9100, -9100, -9100, -9100, -9100, -9100};
        const int32_t xdrop_threshold    = 910;
        const int32_t device_id          = 0;
        const bool no_entropy            = false;
        const std::size_t max_gpu_memory = cudautils::find_largest_contiguous_device_memory_section();
        allocator_                       = create_default_device_allocator(max_gpu_memory);
        stream_                          = make_cuda_stream();
        ungapped_extender_               = create_extender(score_matrix, NUC2, xdrop_threshold, no_entropy, stream_.get(), device_id, allocator_);
    }

    void TearDown()
    {
        ungapped_extender_.reset();
    }

protected:
    std::unique_ptr<Extender> ungapped_extender_;
    End2EndTestParam param_;
    CudaStream stream_;
    DefaultDeviceAllocator allocator_;
};

TEST_P(TestCudaextenderEnd2End, TestCorrectness)
{
    std::unique_ptr<io::FastaParser> fasta_parser_target = io::create_kseq_fasta_parser(param_.target_file, 0, false);
    const std::string target_sequence                    = fasta_parser_target->get_sequence_by_id(0).seq;
    std::unique_ptr<io::FastaParser> fasta_parser_query  = io::create_kseq_fasta_parser(param_.query_file, 0, false);
    const std::string query_sequence                     = fasta_parser_query->get_sequence_by_id(0).seq;
    std::vector<SeedPair> h_seed_pairs;
    parse_seed_pairs(param_.seed_pairs_file, h_seed_pairs);
    // Allocate pinned memory for query and target strings
    pinned_host_vector<int8_t> h_encoded_target(target_sequence.length());
    pinned_host_vector<int8_t> h_encoded_query(target_sequence.length());
    encode_sequence(h_encoded_target.data(), target_sequence.c_str(), target_sequence.length());
    encode_sequence(h_encoded_query.data(), query_sequence.c_str(), query_sequence.length());
    ungapped_extender_->extend_async(h_encoded_query.data(),
                                     get_size<int32_t>(h_encoded_query),
                                     h_encoded_target.data(),
                                     get_size<int32_t>(h_encoded_target),
                                     param_.score_threshold,
                                     h_seed_pairs);
    // Parse golden scored_segment_pairs while extension is in progress
    std::vector<ScoredSegmentPair> golden_scored_segment_pairs;
    parse_scored_segment_pairs(param_.golden_scored_segment_pairs_file, golden_scored_segment_pairs);
    ungapped_extender_->sync();
    std::vector<ScoredSegmentPair> scored_segment_pairs = ungapped_extender_->get_scored_segment_pairs();
    ASSERT_EQ(golden_scored_segment_pairs, scored_segment_pairs);
}

INSTANTIATE_TEST_SUITE_P(TestEnd2End, TestCudaextenderEnd2End, ::testing::ValuesIn(getCudaextenderEnd2EndTestCases()));

} // namespace cudaextender

} // namespace genomeworks

} // namespace claraparabricks
