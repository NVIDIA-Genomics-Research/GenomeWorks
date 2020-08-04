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

#include <claraparabricks/genomeworks/cudapoa/batch.hpp>
#include <claraparabricks/genomeworks/utils/genomeutils.hpp>
#include <claraparabricks/genomeworks/utils/graph.hpp>

#include "gtest/gtest.h"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

using ::testing::TestWithParam;
using ::testing::ValuesIn;

class GraphTest : public ::testing::Test
{
public:
    void SetUp() {}

    void initialize(uint32_t max_sequences_per_poa,
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
        BatchConfig batch_size(1024, max_sequences_per_poa);

        cudapoa_batch = genomeworks::cudapoa::create_batch(device_id, stream, mem_per_batch, output_mask, batch_size, gap_score, mismatch_score, match_score);
    }

public:
    std::unique_ptr<genomeworks::cudapoa::Batch> cudapoa_batch;
};

TEST_F(GraphTest, CudapoaSerializeGraph)
{
    std::minstd_rand rng(1);
    int num_sequences    = 500;
    std::string backbone = genomeworks::genomeutils::generate_random_genome(50, rng);
    auto sequences       = genomeworks::genomeutils::generate_random_sequences(backbone, num_sequences, rng, 10, 5, 10);

    initialize(num_sequences);
    Group poa_group;
    std::vector<StatusType> status;
    std::vector<std::vector<int8_t>> weights;
    for (const auto& seq : sequences)
    {
        weights.push_back(std::vector<int8_t>(seq.length(), 1));
        Entry e{};
        e.seq     = seq.c_str();
        e.weights = weights.back().data();
        e.length  = seq.length();
        poa_group.push_back(e);
    }
    ASSERT_EQ(cudapoa_batch->add_poa_group(status, poa_group), StatusType::success);

    std::vector<DirectedGraph> cudapoa_graphs;
    std::vector<StatusType> output_status;

    cudapoa_batch->generate_poa();

    cudapoa_batch->get_graphs(cudapoa_graphs, output_status);
    std::cout << cudapoa_graphs[0].serialize_to_dot() << std::endl;
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
