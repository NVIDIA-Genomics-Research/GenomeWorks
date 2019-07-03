/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <random>

#include "utils/genomeutils.hpp"
#include "cudautils/cudautils.hpp"
#include "cudaaligner/aligner.hpp"

namespace claragenomics
{

namespace cudaaligner
{

static void BM_SingleBatchAlignment(benchmark::State& state)
{
    int32_t alignments_per_batch = state.range(0);
    int32_t genome_size          = state.range(1);

    // Total memory needed for benchmark
    // TODO: Get GPU memory needed for alignment from Aligner object
    const size_t mem_per_alignment = genome_size * 3 * 1024; // 3 KB per unit, emperically calculated.
    const size_t total_mem         = mem_per_alignment * alignments_per_batch;

    // Query free total and free GPU memory.
    size_t free, total;
    CGA_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));

    if (total_mem > free)
    {
        state.SkipWithError("Not enough available memory for config, skipping");
    }

    // Create aligner object
    std::unique_ptr<Aligner> aligner = create_aligner(genome_size,
                                                      genome_size,
                                                      alignments_per_batch,
                                                      AlignmentType::global,
                                                      0,
                                                      0);

    // Generate random sequences
    std::minstd_rand rng(1);
    for (int32_t i = 0; i < alignments_per_batch; i++)
    {
        // TODO: generate genomes with indels as well
        std::string genome_1 = claragenomics::genomeutils::generate_random_genome(genome_size, rng);
        std::string genome_2 = claragenomics::genomeutils::generate_random_genome(genome_size, rng);

        aligner->add_alignment(genome_1.c_str(), genome_1.length(),
                               genome_2.c_str(), genome_2.length());
    }

    // Run alignment repeatedly
    for (auto _ : state)
    {
        aligner->align_all();
        aligner->sync_alignments();
    }
}

// Register the function as a benchmark
BENCHMARK(BM_SingleBatchAlignment)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Ranges({{32, 512}, {500, 10000}});

} // namespace cudaaligner
} // namespace claragenomics

BENCHMARK_MAIN();
