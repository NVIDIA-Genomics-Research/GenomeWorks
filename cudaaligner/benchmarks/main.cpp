/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "aligner_global_ukkonen.hpp"
#include "aligner_global_myers.hpp"
#include "aligner_global_myers_banded.hpp"
#include "aligner_global_hirschberg_myers.hpp"

#include <claragenomics/utils/genomeutils.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/cudaaligner/aligner.hpp>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <random>

namespace claragenomics
{

namespace cudaaligner
{

static void BM_SingleAlignment(benchmark::State& state)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    const int32_t genome_size        = state.range(0);

    // Generate random sequences
    std::minstd_rand rng(1);
    std::string genome_1 = claragenomics::genomeutils::generate_random_genome(genome_size, rng);
    std::string genome_2 = claragenomics::genomeutils::generate_random_sequence(genome_1, rng, genome_size / 30, genome_size / 30, genome_size / 30); // 3*x/30 = 10% difference

    // Create aligner object
    std::unique_ptr<Aligner> aligner = create_aligner(get_size(genome_1),
                                                      get_size(genome_2),
                                                      1,
                                                      AlignmentType::global_alignment,
                                                      allocator,
                                                      0,
                                                      0);
    aligner->add_alignment(genome_1.c_str(), genome_1.length(),
                           genome_2.c_str(), genome_2.length());

    // Run alignment repeatedly
    for (auto _ : state)
    {
        aligner->align_all();
        aligner->sync_alignments();
    }
}

class CudaStream
{
public:
    CudaStream()
    {
        CGA_CU_CHECK_ERR(cudaStreamCreate(&s_));
    }

    ~CudaStream()
    {
        CGA_CU_CHECK_ERR(cudaStreamDestroy(s_));
    }

    inline cudaStream_t& get()
    {
        return s_;
    }

private:
    cudaStream_t s_ = nullptr;
};

template <typename AlignerT>
static void BM_SingleBatchAlignment(benchmark::State& state)
{
    const std::size_t max_gpu_memory = cudautils::find_largest_contiguous_device_memory_section();
    CudaStream stream;
    DefaultDeviceAllocator allocator   = create_default_device_allocator(max_gpu_memory);
    const int32_t alignments_per_batch = state.range(0);
    const int32_t genome_size          = state.range(1);

    std::unique_ptr<Aligner> aligner;
    // Create aligner object
    try
    {
        aligner = std::make_unique<AlignerT>(
            genome_size,
            genome_size,
            alignments_per_batch,
            allocator,
            stream.get(),
            0);

        // Generate random sequences
        std::minstd_rand rng(1);
        for (int32_t i = 0; i < alignments_per_batch; i++)
        {
            // TODO: generate genomes with indels as well
            std::string genome_1 = claragenomics::genomeutils::generate_random_genome(genome_size, rng);
            std::string genome_2 = claragenomics::genomeutils::generate_random_sequence(genome_1, rng, genome_size / 30, genome_size / 30, genome_size / 30); // 3*x/30 = 10% difference
            if (get_size(genome_2) > genome_size)
            {
                genome_2.resize(genome_size);
            }

            aligner->add_alignment(genome_1.c_str(), genome_1.length(),
                                   genome_2.c_str(), genome_2.length());
        }
    }
    catch (device_memory_allocation_exception const& e)
    {
        state.SkipWithError("Could not allocate enough memory for config, skipping");
    }

    // Run alignment repeatedly
    for (auto _ : state)
    {
        aligner->align_all();
        aligner->sync_alignments();
    }
}

// Register the functions as a benchmark
BENCHMARK(BM_SingleAlignment)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(10)
    ->Range(100, 100000);

BENCHMARK_TEMPLATE(BM_SingleBatchAlignment, AlignerGlobalUkkonen)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Ranges({{32, 1024}, {512, 65536}});

BENCHMARK_TEMPLATE(BM_SingleBatchAlignment, AlignerGlobalMyers)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Ranges({{32, 1024}, {512, 65536}});

BENCHMARK_TEMPLATE(BM_SingleBatchAlignment, AlignerGlobalMyersBanded)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Ranges({{32, 1024}, {512, 65536}});

BENCHMARK_TEMPLATE(BM_SingleBatchAlignment, AlignerGlobalHirschbergMyers)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Ranges({{32, 1024}, {512, 65536}});

} // namespace cudaaligner
} // namespace claragenomics

BENCHMARK_MAIN();
