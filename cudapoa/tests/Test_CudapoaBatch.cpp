/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claragenomics/cudapoa/batch.hpp>

#include "gtest/gtest.h"

namespace claragenomics
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

    void initialize(int32_t max_sequences_per_poa,
                    size_t mem_per_batch,
                    int32_t device_id,
                    int8_t output_mask     = OutputType::consensus,
                    int16_t gap_score      = -8,
                    int16_t mismatch_score = -6,
                    int16_t match_score    = 8,
                    bool banded_alignment  = false)
    {
        cudapoa_batch = claragenomics::cudapoa::create_batch(max_sequences_per_poa,
                                                             device_id,
                                                             mem_per_batch,
                                                             output_mask,
                                                             gap_score,
                                                             mismatch_score,
                                                             match_score,
                                                             banded_alignment);
    }

    size_t get_free_device_mem(int32_t device_id)
    {
        size_t total = 0, free = 0;
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free, &total);
        return free;
    }

public:
    std::unique_ptr<claragenomics::cudapoa::Batch> cudapoa_batch;
};

TEST_F(TestCudapoaBatch, InitializeTest)
{
    const int32_t device_id = 0;
    size_t free             = get_free_device_mem(device_id);
    initialize(5, 0.9 * free, device_id);
    EXPECT_EQ(cudapoa_batch->batch_id(), 0);
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 0);
}

TEST_F(TestCudapoaBatch, InitializeZeroMemTest)
{
    const int32_t device_id = 0;
    EXPECT_THROW(initialize(5, 0, device_id), std::runtime_error);
}

TEST_F(TestCudapoaBatch, AddPOATest)
{
    const int32_t device_id = 0;
    size_t free             = get_free_device_mem(device_id);
    initialize(5, 0.9 * free, device_id);
    Group poa_group;
    poa_group.push_back(Entry{});
    std::vector<StatusType> status;
    StatusType call_status = cudapoa_batch->add_poa_group(status, poa_group);
    EXPECT_EQ(call_status, StatusType::success) << static_cast<int32_t>(call_status);
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 1);
    cudapoa_batch->reset();
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 0);
}

TEST_F(TestCudapoaBatch, MaxSeqPerPOATest)
{
    const int32_t device_id = 0;
    size_t free             = get_free_device_mem(device_id);
    initialize(10, 0.9 * free, device_id);
    Group poa_group;
    std::vector<StatusType> status;

    int32_t seq_length = 20;
    std::string seq(seq_length, 'A');
    std::vector<int8_t> weights(seq_length, 1);
    for (uint16_t i = 0; i < 10; ++i)
    {
        Entry e{};
        e.seq     = seq.c_str();
        e.weights = weights.data();
        e.length  = seq.length();
        poa_group.push_back(e);
    }
    EXPECT_EQ(cudapoa_batch->add_poa_group(status, poa_group), StatusType::success);
    EXPECT_EQ(cudapoa_batch->get_total_poas(), 1);
    EXPECT_EQ(status.at(9), StatusType::exceeded_maximum_sequences_per_poa);
}

TEST_F(TestCudapoaBatch, MaxSeqSizeTest)
{
    const int32_t device_id = 0;
    size_t free             = get_free_device_mem(device_id);
    initialize(10, 0.9 * free, device_id);
    Group poa_group;
    std::vector<StatusType> status;
    Entry e{};

    int32_t seq_length = 1023;
    std::string seq(seq_length, 'A');
    std::vector<int8_t> weights(seq_length, 1);
    e.seq     = seq.c_str();
    e.weights = weights.data();
    e.length  = seq.length();
    poa_group.push_back(e);

    Entry e_2{};
    seq_length        = 1024;
    std::string seq_2 = std::string(seq_length, 'A');
    std::vector<int8_t> weights_2(seq_length, 1);
    e_2.seq     = seq_2.c_str();
    e_2.weights = weights_2.data();
    e_2.length  = seq_2.length();
    poa_group.push_back(e_2);

    EXPECT_EQ(cudapoa_batch->add_poa_group(status, poa_group), StatusType::success);
    EXPECT_EQ(status.at(1), StatusType::exceeded_maximum_sequence_size);
}

} // namespace cudapoa

} // namespace claragenomics
