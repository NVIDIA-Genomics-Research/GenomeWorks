/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <assert.h>
#include <algorithm>
#include "gtest/gtest.h"
#include "cudapoa/batch.hpp"
#include <utils/genomeutils.hpp>
#include "spoa/spoa.hpp"

namespace claragenomics
{

namespace cudapoa
{

using ::testing::TestWithParam;
using ::testing::ValuesIn;

class MSATest : public ::testing::Test
{
public:
    void SetUp() {}

    void initialize(uint32_t max_poas,
                    uint32_t max_sequences_per_poa,
                    cudaStream_t stream    = 0,
                    uint32_t device_id     = 0,
                    int8_t output_mask     = OutputType::msa,
                    int16_t gap_score      = -8,
                    int16_t mismatch_score = -6,
                    int16_t match_score    = 8)
    {
        cudapoa_batch = claragenomics::cudapoa::create_batch(max_poas, max_sequences_per_poa, stream, device_id, output_mask, gap_score, mismatch_score, match_score);
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
    std::unique_ptr<claragenomics::cudapoa::Batch> cudapoa_batch;
};

TEST_F(MSATest, CudapoaMSA)
{
    std::minstd_rand rng(1);
    int num_sequences    = 499;
    std::string backbone = claragenomics::genomeutils::generate_random_genome(50, rng);
    auto sequences       = claragenomics::genomeutils::generate_random_sequences(backbone, num_sequences, rng, 10, 5, 10);

    initialize(1, num_sequences + 1); //
    EXPECT_EQ(cudapoa_batch->add_poa(), StatusType::success);
    for (const auto& seq : sequences)
    {
        EXPECT_EQ(cudapoa_batch->add_seq_to_poa(seq.c_str(), nullptr, seq.length()), StatusType::success); //nullptr for weights will make it all 1
    }

    std::vector<std::vector<std::string>> cudapoa_msa;
    std::vector<StatusType> output_status;

    cudapoa_batch->generate_poa();

    cudapoa_batch->get_msa(cudapoa_msa, output_status);

    auto spoa_msa = spoa_generate_multiple_sequence_alignments(sequences);

#ifndef SPOA_ACCURATE
    for (int i = 0; i < spoa_msa.size(); i++)
    {

        std::string msa = cudapoa_msa[0][i];
        msa.erase(std::remove(msa.begin(), msa.end(), '-'), msa.end());
        EXPECT_EQ(msa, sequences[i]);
    }
#else
    EXPECT_EQ(spoa_msa, cudapoa_msa[0]);
#endif
}

} // namespace cudapoa

} // namespace claragenomics
