/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include <exception>

#include "../src/index_batcher.cuh"

#include "cudamapper_file_location.hpp"

#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claragenomics
{
namespace cudamapper
{

void test_generated_batches(const std::vector<BatchOfIndices>& generated_batches,
                            const std::vector<BatchOfIndices>& expected_batches)
{
    ASSERT_EQ(expected_batches.size(), generated_batches.size());

    for (std::int64_t i = 0; i < get_size<std::int64_t>(expected_batches); ++i)
    {
        const BatchOfIndices& expected_batch                                   = expected_batches[i];
        const BatchOfIndices& generated_batch                                  = generated_batches[i];
        const std::vector<IndexDescriptor>& expected_batch_host_query_indices  = expected_batch.host_batch.query_indices;
        const std::vector<IndexDescriptor>& generated_batch_host_query_indices = generated_batch.host_batch.query_indices;
        ASSERT_EQ(get_size<std::int64_t>(expected_batch_host_query_indices), get_size<std::int64_t>(generated_batch_host_query_indices)) << "i: " << i;
        for (std::int64_t j = 0; j < get_size<std::int64_t>(expected_batch_host_query_indices); ++j)
        {
            const IndexDescriptor& expected_host_query_index  = expected_batch_host_query_indices[j];
            const IndexDescriptor& generated_host_query_index = generated_batch_host_query_indices[j];
            ASSERT_EQ(expected_host_query_index.first_read(), generated_host_query_index.first_read()) << "i: " << i << ", j: " << j;
            ASSERT_EQ(expected_host_query_index.number_of_reads(), generated_host_query_index.number_of_reads()) << "i: " << i << ", j: " << j;
        }
        const std::vector<IndexDescriptor>& expected_batch_host_target_indices  = expected_batch.host_batch.target_indices;
        const std::vector<IndexDescriptor>& generated_batch_host_target_indices = generated_batch.host_batch.target_indices;
        ASSERT_EQ(get_size<std::int64_t>(expected_batch_host_target_indices), get_size<std::int64_t>(generated_batch_host_target_indices)) << "i: " << i;
        for (std::int64_t j = 0; j < get_size<std::int64_t>(expected_batch_host_target_indices); ++j)
        {
            const IndexDescriptor& expected_host_target_index  = expected_batch_host_target_indices[j];
            const IndexDescriptor& generated_host_target_index = generated_batch_host_target_indices[j];
            ASSERT_EQ(expected_host_target_index.first_read(), generated_host_target_index.first_read()) << "i: " << i << ", j: " << j;
            ASSERT_EQ(expected_host_target_index.number_of_reads(), generated_host_target_index.number_of_reads()) << "i: " << i << ", j: " << j;
        }

        ASSERT_EQ(get_size<std::int64_t>(expected_batch.device_batches), get_size<std::int64_t>(generated_batch.device_batches)) << "i: " << i;
        for (std::int64_t j = 0; j < get_size<std::int64_t>(expected_batch.device_batches); ++j)
        {
            const IndexBatch& expected_device_batch                            = expected_batch.device_batches[j];
            const IndexBatch& generated_device_batch                           = generated_batch.device_batches[j];
            const std::vector<IndexDescriptor>& expected_device_query_indices  = expected_device_batch.query_indices;
            const std::vector<IndexDescriptor>& generated_device_query_indices = generated_device_batch.query_indices;
            ASSERT_EQ(get_size<std::int64_t>(expected_device_query_indices), get_size<std::int64_t>(generated_device_query_indices)) << "i: " << i << ", j: " << j;
            for (std::int64_t k = 0; k < get_size<std::int64_t>(expected_device_query_indices); ++k)
            {
                const IndexDescriptor& expected_device_query_index  = expected_device_query_indices[k];
                const IndexDescriptor& generated_device_query_index = generated_device_query_indices[k];
                ASSERT_EQ(expected_device_query_index.first_read(), generated_device_query_index.first_read()) << "i: " << i << ", j: " << j << ", k: " << k;
                ASSERT_EQ(expected_device_query_index.number_of_reads(), generated_device_query_index.number_of_reads()) << "i: " << i << ", j: " << j << ", k: " << k;
            }
            const std::vector<IndexDescriptor>& expected_device_target_indices  = expected_device_batch.target_indices;
            const std::vector<IndexDescriptor>& generated_device_target_indices = generated_device_batch.target_indices;
            ASSERT_EQ(get_size<std::int64_t>(expected_device_target_indices), get_size<std::int64_t>(generated_device_target_indices)) << "i: " << i << ", j: " << j;
            for (std::int64_t k = 0; k < get_size<std::int64_t>(expected_device_target_indices); ++k)
            {
                const IndexDescriptor& expected_device_target_index  = expected_device_target_indices[k];
                const IndexDescriptor& generated_device_target_index = generated_device_target_indices[k];
                ASSERT_EQ(expected_device_target_index.first_read(), generated_device_target_index.first_read()) << "i: " << i << ", j: " << j << ", k: " << k;
                ASSERT_EQ(expected_device_target_index.number_of_reads(), generated_device_target_index.number_of_reads()) << "i: " << i << ", j: " << j << ", k: " << k;
            }
        }
    }
}

// *** test generate_batches_of_indices ***

void test_generate_batches_of_indices(const number_of_indices_t query_indices_per_host_batch,
                                      const number_of_indices_t query_indices_per_device_batch,
                                      const number_of_indices_t target_indices_per_host_batch,
                                      const number_of_indices_t target_indices_per_device_batch,
                                      const std::shared_ptr<const claragenomics::io::FastaParser> query_parser,
                                      const std::shared_ptr<const claragenomics::io::FastaParser> target_parser,
                                      const number_of_basepairs_t query_basepairs_per_index,
                                      const number_of_basepairs_t target_basepairs_per_index,
                                      const bool same_query_and_target,
                                      const std::vector<BatchOfIndices>& expected_batches)
{
    const std::vector<BatchOfIndices>& generated_batches = generate_batches_of_indices(query_indices_per_host_batch,
                                                                                       query_indices_per_device_batch,
                                                                                       target_indices_per_host_batch,
                                                                                       target_indices_per_device_batch,
                                                                                       query_parser,
                                                                                       target_parser,
                                                                                       query_basepairs_per_index,
                                                                                       target_basepairs_per_index,
                                                                                       same_query_and_target);

    test_generated_batches(generated_batches,
                           expected_batches);
}

TEST(TestCudamapperIndexBatcher, test_generate_batches_of_indices_query_and_target_not_the_same)
{
    // query FASTA: 10_reads.fasta
    // read_0: 5 basepairs
    // read_1: 2 basepairs
    // read_2: 2 basepairs
    // read_3: 2 basepairs
    // read_4: 2 basepairs
    // read_5: 3 basepairs
    // read_6: 7 basepairs
    // read_7: 2 basepairs
    // read_8: 8 basepairs
    // read_9: 3 basepairs
    // max basepairs in index: 10
    // indices: {0, 3}, {3, 3}, {6, 2}, {8, 1}, {9, 1}
    // query_indices_per_host_batch: 2
    // query_indices_per_device_batch: 1
    // host batches: {{0, 3}, {3, 3}}, {{6, 2}, {8, 1}}, {9, 1}
    // device bathes: {{0, 3}}, {{3, 3}}
    //                {{6, 2}}, {{8, 1}}
    //                {{9, 1}}

    // target FASTA: 20_reads.fasta
    // read_0:  4 basepairs
    // read_1:  6 basepairs
    // read_2:  7 basepairs
    // read_3:  4 basepairs
    // read_4:  3 basepairs
    // read_5:  8 basepairs
    // read_6:  6 basepairs
    // read_7:  3 basepairs
    // read_8:  3 basepairs
    // read_9:  5 basepairs
    // read_10: 7 basepairs
    // read_11: 3 basepairs
    // read_12: 2 basepairs
    // read_13: 4 basepairs
    // read_14: 4 basepairs
    // read_15: 2 basepairs
    // read_16: 5 basepairs
    // read_17: 6 basepairs
    // read_18: 2 basepairs
    // read_19: 4 basepairs
    // max basepairs in index: 10
    // indices: {0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}, {8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}, {19, 1}
    // query_indices_per_host_batch: 5
    // query_indices_per_device_batch: 2
    // host batches: {{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}}, {{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}}, {{19, 1}}
    // device bathes: {{0, 2}, {2, 1}}, {{3, 2}, {5, 1}}, {{6, 2}}
    //                {{8, 2}, {10, 2}}, {{12, 3}, {15, 2}}, {{17, 2}}
    //                {{19, 1}}

    const number_of_indices_t query_indices_per_host_batch                    = 2;
    const number_of_indices_t query_indices_per_device_batch                  = 1;
    const number_of_indices_t target_indices_per_host_batch                   = 5;
    const number_of_indices_t target_indices_per_device_batch                 = 2;
    const std::shared_ptr<const claragenomics::io::FastaParser> query_parser  = claragenomics::io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/10_reads.fasta", 1, false);
    const std::shared_ptr<const claragenomics::io::FastaParser> target_parser = claragenomics::io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta", 1, false);
    const number_of_basepairs_t query_basepairs_per_index                     = 10;
    const number_of_basepairs_t target_basepairs_per_index                    = 10;
    const bool same_query_and_target                                          = false;

    std::vector<BatchOfIndices> expected_batches;
    {
        // query 0, target 0
        std::vector<IndexDescriptor> host_batch_query_indices{{0, 3}, {3, 3}};
        std::vector<IndexDescriptor> host_batch_target_indices{{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{0, 2}, {2, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{3, 2}, {5, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{6, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{0, 2}, {2, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{3, 2}, {5, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{6, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 0, target 1
        std::vector<IndexDescriptor> host_batch_query_indices{{0, 3}, {3, 3}};
        std::vector<IndexDescriptor> host_batch_target_indices{{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 0, target 2
        std::vector<IndexDescriptor> host_batch_query_indices{{0, 3}, {3, 3}};
        std::vector<IndexDescriptor> host_batch_target_indices{{19, 1}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 3}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 1, target 0
        std::vector<IndexDescriptor> host_batch_query_indices{{6, 2}, {8, 1}};
        std::vector<IndexDescriptor> host_batch_target_indices{{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{0, 2}, {2, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{3, 2}, {5, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{6, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{0, 2}, {2, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{3, 2}, {5, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{6, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 1, target 1
        std::vector<IndexDescriptor> host_batch_query_indices{{6, 2}, {8, 1}};
        std::vector<IndexDescriptor> host_batch_target_indices{{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 1, target 2
        std::vector<IndexDescriptor> host_batch_query_indices{{6, 2}, {8, 1}};
        std::vector<IndexDescriptor> host_batch_target_indices{{19, 1}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
            //    device_batches.push_back({});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
            //    device_batches.push_back({});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 2, target 0
        std::vector<IndexDescriptor> host_batch_query_indices{{9, 1}};
        std::vector<IndexDescriptor> host_batch_target_indices{{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{9, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{0, 2}, {2, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{9, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{3, 2}, {5, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{9, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{6, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 2, target 1
        std::vector<IndexDescriptor> host_batch_query_indices{{9, 1}};
        std::vector<IndexDescriptor> host_batch_target_indices{{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{9, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{9, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{9, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 2, target 2
        std::vector<IndexDescriptor> host_batch_query_indices{{9, 1}};
        std::vector<IndexDescriptor> host_batch_target_indices{{19, 1}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{9, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }

    test_generate_batches_of_indices(query_indices_per_host_batch,
                                     query_indices_per_device_batch,
                                     target_indices_per_host_batch,
                                     target_indices_per_device_batch,
                                     query_parser,
                                     target_parser,
                                     query_basepairs_per_index,
                                     target_basepairs_per_index,
                                     same_query_and_target,
                                     expected_batches);
}

TEST(TestCudamapperIndexBatcher, test_generate_batches_of_indices_same_query_and_target)
{
    // target FASTA: 20_reads.fasta
    // read_0:  4 basepairs
    // read_1:  6 basepairs
    // read_2:  7 basepairs
    // read_3:  4 basepairs
    // read_4:  3 basepairs
    // read_5:  8 basepairs
    // read_6:  6 basepairs
    // read_7:  3 basepairs
    // read_8:  3 basepairs
    // read_9:  5 basepairs
    // read_10: 7 basepairs
    // read_11: 3 basepairs
    // read_12: 2 basepairs
    // read_13: 4 basepairs
    // read_14: 4 basepairs
    // read_15: 2 basepairs
    // read_16: 5 basepairs
    // read_17: 6 basepairs
    // read_18: 2 basepairs
    // read_19: 4 basepairs
    // max basepairs in index: 10
    // indices: {0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}, {8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}, {19, 1}
    // query_indices_per_host_batch: 5
    // query_indices_per_device_batch: 2
    // host batches: {{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}}, {{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}}, {{19, 1}}
    // device bathes: {{0, 2}, {2, 1}}, {{3, 2}, {5, 1}}, {{6, 2}}
    //                {{8, 2}, {10, 2}}, {{12, 3}, {15, 2}}, {{17, 2}}
    //                {{19, 1}}

    const number_of_indices_t query_indices_per_host_batch                    = 5;
    const number_of_indices_t query_indices_per_device_batch                  = 2;
    const number_of_indices_t target_indices_per_host_batch                   = 5;
    const number_of_indices_t target_indices_per_device_batch                 = 2;
    const std::shared_ptr<const claragenomics::io::FastaParser> query_parser  = claragenomics::io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta", 1, false);
    const std::shared_ptr<const claragenomics::io::FastaParser> target_parser = query_parser;
    const number_of_basepairs_t query_basepairs_per_index                     = 10;
    const number_of_basepairs_t target_basepairs_per_index                    = 10;
    const bool same_query_and_target                                          = true;

    std::vector<BatchOfIndices> expected_batches;
    {
        // query 0, target 0
        std::vector<IndexDescriptor> host_batch_query_indices{{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}};
        std::vector<IndexDescriptor> host_batch_target_indices{{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 2}, {2, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{0, 2}, {2, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 2}, {2, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{3, 2}, {5, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 2}, {2, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{6, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        // skipping device query 1, device target 0 due to symmetry
        {
            // device query 1, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 2}, {5, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{3, 2}, {5, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 2}, {5, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{6, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        // skipping device query 2, device target 0 due to symmetry
        // skipping device query 2, device target 1 due to symmetry
        {
            // device query 2, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{6, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 0, target 1
        std::vector<IndexDescriptor> host_batch_query_indices{{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}};
        std::vector<IndexDescriptor> host_batch_target_indices{{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 2}, {2, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 2}, {2, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 2}, {2, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 2}, {5, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 2}, {5, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 2}, {5, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 2, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 2, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 2, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 0, target 2
        std::vector<IndexDescriptor> host_batch_query_indices{{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}};
        std::vector<IndexDescriptor> host_batch_target_indices{{19, 1}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{0, 2}, {2, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{3, 2}, {5, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 2, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{6, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    // skipping query 1, target 0 due to symmetry
    {
        // query 1, target 1
        std::vector<IndexDescriptor> host_batch_query_indices{{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}};
        std::vector<IndexDescriptor> host_batch_target_indices{{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 2}, {10, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{8, 2}, {10, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 2}, {10, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 0, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 2}, {10, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        // skipping device query 1, device target 0 due to symmetry
        {
            // device query 1, device target 1
            std::vector<IndexDescriptor> device_batch_query_indices{{12, 3}, {15, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{12, 3}, {15, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{12, 3}, {15, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        // skipping device query 2, device target 0 due to symmetry
        // skipping device query 2, device target 1 due to symmetry
        {
            // device query 2, device target 2
            std::vector<IndexDescriptor> device_batch_query_indices{{17, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{17, 2}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    {
        // query 1, target 2
        std::vector<IndexDescriptor> host_batch_query_indices{{8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}};
        std::vector<IndexDescriptor> host_batch_target_indices{{19, 1}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{8, 2}, {10, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 1, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{12, 3}, {15, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        {
            // device query 2, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{17, 2}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }
    // skipping query 2, target 0 due to symmetry
    // skipping query 2, target 1 due to symmetry
    {
        // query 2, target 2
        std::vector<IndexDescriptor> host_batch_query_indices{{19, 1}};
        std::vector<IndexDescriptor> host_batch_target_indices{{19, 1}};
        IndexBatch host_batch{std::move(host_batch_query_indices),
                              std::move(host_batch_target_indices)};
        std::vector<IndexBatch> device_batches;
        {
            // device query 0, device target 0
            std::vector<IndexDescriptor> device_batch_query_indices{{19, 1}};
            std::vector<IndexDescriptor> device_batch_target_indices{{19, 1}};
            device_batches.push_back({std::move(device_batch_query_indices),
                                      std::move(device_batch_target_indices)});
        }
        expected_batches.push_back({std::move(host_batch),
                                    std::move(device_batches)});
    }

    test_generate_batches_of_indices(query_indices_per_host_batch,
                                     query_indices_per_device_batch,
                                     target_indices_per_host_batch,
                                     target_indices_per_device_batch,
                                     query_parser,
                                     target_parser,
                                     query_basepairs_per_index,
                                     target_basepairs_per_index,
                                     same_query_and_target,
                                     expected_batches);
}

TEST(TestCudamapperIndexBatcher, test_generate_batches_of_indices_exceptions)
{
    const number_of_indices_t query_indices_per_host_batch                   = 5;
    const number_of_indices_t query_indices_per_device_batch                 = 2;
    const number_of_basepairs_t query_basepairs_per_index                    = 10;
    const std::shared_ptr<const claragenomics::io::FastaParser> query_parser = claragenomics::io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/10_reads.fasta", 1, false);

    number_of_indices_t target_indices_per_host_batch                   = query_indices_per_host_batch;
    number_of_indices_t target_indices_per_device_batch                 = query_indices_per_device_batch;
    number_of_basepairs_t target_basepairs_per_index                    = query_basepairs_per_index;
    std::shared_ptr<const claragenomics::io::FastaParser> target_parser = query_parser;

    const bool same_query_and_target = true;

    // indices_per_host_batch different
    target_indices_per_host_batch = 100;
    ASSERT_THROW(generate_batches_of_indices(query_indices_per_host_batch,
                                             query_indices_per_device_batch,
                                             target_indices_per_host_batch,
                                             target_indices_per_device_batch,
                                             query_parser,
                                             target_parser,
                                             query_basepairs_per_index,
                                             target_basepairs_per_index,
                                             same_query_and_target),
                 std::invalid_argument);
    target_indices_per_host_batch = query_indices_per_host_batch;

    // indices_per_device_batch different
    target_indices_per_device_batch = 100;
    ASSERT_THROW(generate_batches_of_indices(query_indices_per_host_batch,
                                             query_indices_per_device_batch,
                                             target_indices_per_host_batch,
                                             target_indices_per_device_batch,
                                             query_parser,
                                             target_parser,
                                             query_basepairs_per_index,
                                             target_basepairs_per_index,
                                             same_query_and_target),
                 std::invalid_argument);
    target_indices_per_device_batch = query_indices_per_device_batch;

    // parser different
    target_parser = claragenomics::io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta", 1, false);
    ASSERT_THROW(generate_batches_of_indices(query_indices_per_host_batch,
                                             query_indices_per_device_batch,
                                             target_indices_per_host_batch,
                                             target_indices_per_device_batch,
                                             query_parser,
                                             target_parser,
                                             query_basepairs_per_index,
                                             target_basepairs_per_index,
                                             same_query_and_target),
                 std::invalid_argument);
    target_parser = query_parser;

    // indices_per_host_batch different
    target_basepairs_per_index = 100;
    ASSERT_THROW(generate_batches_of_indices(query_indices_per_host_batch,
                                             query_indices_per_device_batch,
                                             target_indices_per_host_batch,
                                             target_indices_per_device_batch,
                                             query_parser,
                                             target_parser,
                                             query_basepairs_per_index,
                                             target_basepairs_per_index,
                                             same_query_and_target),
                 std::invalid_argument);
    target_basepairs_per_index = query_basepairs_per_index;
}

} // namespace cudamapper
} // namespace claragenomics
