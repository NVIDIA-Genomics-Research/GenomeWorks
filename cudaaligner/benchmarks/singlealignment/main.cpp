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

#include "utils/genomeutils.hpp"
#include "cudautils/cudautils.hpp"
#include "cudaaligner/aligner.hpp"

namespace genomeworks
{

namespace cudaaligner
{

static void BM_SingleAlignment(benchmark::State& state)
{
    int32_t genome_size = state.range(0);

    // Generate random sequences
    std::minstd_rand rng(1);
    std::string genome_1 = genomeworks::genomeutils::generate_random_genome(genome_size, rng);
    std::string genome_2 = genomeworks::genomeutils::generate_random_genome(genome_size, rng);

    // Create aligner object
    std::unique_ptr<Aligner> aligner = create_aligner(genome_size,
                                                      genome_size,
                                                      1,
                                                      AlignmentType::global,
                                                      0);
    aligner->set_cuda_stream(0);
    aligner->add_alignment(genome_1.c_str(), genome_1.length(),
                           genome_2.c_str(), genome_2.length());

    // Run alignment repeatedly
    for (auto _ : state)
    {
        aligner->align_all();
        aligner->sync_alignments();
    }
}

// Register the function as a benchmark
BENCHMARK(BM_SingleAlignment)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(100, 1000000);
} // namespace cudaaligner
} // namespace genomeworks

BENCHMARK_MAIN();
