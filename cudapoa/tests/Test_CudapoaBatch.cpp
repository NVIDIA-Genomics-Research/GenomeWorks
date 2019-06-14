/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"
#include "cudapoa/batch.hpp"

namespace genomeworks
{

namespace cudapoa
{

class TestCudapoaBatch : public ::testing::Test
{
public:
    void SetUp()
    {
        // Do noting for now, but place for
        // constructing test objects.
    }

    void initialize(int32_t max_poas,
                    int32_t max_sequences_per_poa,
                    int32_t device_id      = 0,
                    int16_t gap_score      = -8,
                    int16_t mismatch_score = -6,
                    int16_t match_score    = 8)
    {
        cudapoa_batch = genomeworks::cudapoa::create_batch(max_poas, max_sequences_per_poa, device_id, gap_score, mismatch_score, match_score);
    }

public:
    std::unique_ptr<genomeworks::cudapoa::Batch> cudapoa_batch;
};

TEST_F(TestCudapoaBatch, InitializeTest)
{
    initialize(5, 5);
    EXPECT_EQ(cudapoa_batch->batch_id(), 0);
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 0);
}

TEST_F(TestCudapoaBatch, AddPOATest)
{
    initialize(5, 5);
    EXPECT_EQ(cudapoa_batch->add_poa(), StatusType::success);
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 1);
    cudapoa_batch->reset();
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 0);
}

TEST_F(TestCudapoaBatch, MaxPOATest)
{
    initialize(5, 5);

    for (uint16_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(cudapoa_batch->add_poa(), StatusType::success);
    }
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 5);
    EXPECT_EQ(cudapoa_batch->add_poa(), StatusType::exceeded_maximum_poas);
}

TEST_F(TestCudapoaBatch, MaxSeqPerPOATest)
{
    initialize(5, 10);
    EXPECT_EQ(cudapoa_batch->add_poa(), StatusType::success);

    int32_t seq_length = 20;
    std::string seq(seq_length, 'A');
    std::vector<int8_t> weights(seq_length, 1);
    for (uint16_t i = 0; i < 9; ++i)
    {
        EXPECT_EQ(cudapoa_batch->add_seq_to_poa(seq.c_str(), weights.data(), seq.length()), StatusType::success);
    }
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 1);
    EXPECT_EQ(cudapoa_batch->add_seq_to_poa(seq.c_str(), weights.data(), seq.length()), StatusType::exceeded_maximum_sequences_per_poa);
}

TEST_F(TestCudapoaBatch, MaxSeqSizeTest)
{
    initialize(5, 10);
    EXPECT_EQ(cudapoa_batch->add_poa(), StatusType::success);
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 1);

    int32_t seq_length = 1023;
    std::string seq(seq_length, 'A');
    std::vector<int8_t> weights(seq_length, 1);
    EXPECT_EQ(cudapoa_batch->add_seq_to_poa(seq.c_str(), weights.data(), seq.length()), StatusType::success);

    seq_length = 1024;
    seq        = std::string(seq_length, 'A');
    std::vector<int8_t> weights_2(seq_length, 1);
    EXPECT_EQ(cudapoa_batch->add_seq_to_poa(seq.c_str(), weights_2.data(), seq.length()), StatusType::exceeded_maximum_sequence_size);
}

} // namespace cudapoa

} // namespace genomeworks
