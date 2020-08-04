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

#include "../src/cudapoa_kernels.cuh"

#include <claraparabricks/genomeworks/cudapoa/batch.hpp>
#include <claraparabricks/genomeworks/utils/genomeutils.hpp>

#include "gtest/gtest.h"
#include "spoa/spoa.hpp"
#include <assert.h>
#include <algorithm>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

using ::testing::TestWithParam;
using ::testing::ValuesIn;

class MSATest : public ::testing::Test
{
public:
    void SetUp() {}

    void initialize(const BatchConfig& batch_size,
                    uint32_t device_id     = 0,
                    cudaStream_t stream    = 0,
                    int8_t output_mask     = OutputType::msa,
                    int16_t gap_score      = -8,
                    int16_t mismatch_score = -6,
                    int16_t match_score    = 8)
    {
        size_t total = 0, free = 0;
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free, &total);
        size_t mem_per_batch = 0.9 * free;

        cudapoa_batch = genomeworks::cudapoa::create_batch(device_id, stream, mem_per_batch, output_mask, batch_size, gap_score, mismatch_score, match_score);
    }

    std::vector<std::string> spoa_generate_multiple_sequence_alignments(std::vector<std::string> sequences,
                                                                        spoa::AlignmentType atype = spoa::AlignmentType::kNW,
                                                                        int match_score           = 8,
                                                                        int mismatch_score        = -6,
                                                                        int gap_score             = -8)
    {
        auto alignment_engine = spoa::createAlignmentEngine(atype, match_score, mismatch_score, gap_score);
        auto graph            = spoa::createGraph();

        for (const auto& it : sequences)
        {
            auto alignment = alignment_engine->align(it, graph);
            graph->add_alignment(alignment, it);
        }

        std::vector<std::string> msa;
        graph->generate_multiple_sequence_alignment(msa);

        return msa;
    }

public:
    std::unique_ptr<genomeworks::cudapoa::Batch> cudapoa_batch;
};

TEST_F(MSATest, CudapoaMSA)
{
    std::minstd_rand rng(1);
    int num_sequences = 500;
    BatchConfig batch_size(1024, num_sequences);

    std::string backbone = genomeworks::genomeutils::generate_random_genome(50, rng);
    auto sequences       = genomeworks::genomeutils::generate_random_sequences(backbone, num_sequences, rng, 10, 5, 10);

    initialize(batch_size);
    Group poa_group;
    std::vector<StatusType> status;
    for (const auto& seq : sequences)
    {
        Entry e{};
        e.seq     = seq.c_str();
        e.weights = nullptr;
        e.length  = seq.length();
        poa_group.push_back(e);
    }
    ASSERT_EQ(cudapoa_batch->add_poa_group(status, poa_group), StatusType::success);

    std::vector<std::vector<std::string>> cudapoa_msa;
    std::vector<StatusType> output_status;

    cudapoa_batch->generate_poa();

    cudapoa_batch->get_msa(cudapoa_msa, output_status);

    ASSERT_EQ(output_status[0], StatusType::success);
    ASSERT_EQ(poa_group.size(), cudapoa_msa[0].size());

    auto spoa_msa = spoa_generate_multiple_sequence_alignments(sequences);

#ifndef SPOA_ACCURATE
    for (uint32_t i = 0; i < spoa_msa.size(); i++)
    {

        std::string msa = cudapoa_msa[0][i];
        msa.erase(std::remove(msa.begin(), msa.end(), '-'), msa.end());
        ASSERT_EQ(msa, sequences[i]);
    }
#else
    ASSERT_EQ(spoa_msa, cudapoa_msa[0]);
#endif
}

TEST_F(MSATest, CudapoaMSAFailure)
{
    std::minstd_rand rng(1);
    int num_sequences = 10;
    BatchConfig batch_size(1024, num_sequences);
    batch_size.max_consensus_size = batch_size.max_sequence_size;

    std::string backbone = genomeworks::genomeutils::generate_random_genome(batch_size.max_consensus_size - 1, rng);
    auto sequences       = genomeworks::genomeutils::generate_random_sequences(backbone, num_sequences, rng, 10, 5, 10);

    initialize(batch_size);
    Group poa_group;
    std::vector<StatusType> status;
    for (const auto& seq : sequences)
    {
        Entry e{};
        e.seq     = seq.c_str();
        e.weights = nullptr;
        e.length  = seq.length();
        poa_group.push_back(e);
    }
    ASSERT_EQ(cudapoa_batch->add_poa_group(status, poa_group), StatusType::success);

    std::vector<std::vector<std::string>> cudapoa_msa;
    std::vector<StatusType> output_status;

    cudapoa_batch->generate_poa();

    cudapoa_batch->get_msa(cudapoa_msa, output_status);

    ASSERT_EQ(output_status[0], StatusType::exceeded_maximum_sequence_size);
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
