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

#include "aligner_global_ukkonen.hpp"
#include "aligner_global_myers.hpp"
#include "aligner_global_myers_banded.hpp"
#include "aligner_global_hirschberg_myers.hpp"

#include <claraparabricks/genomeworks/utils/genomeutils.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <random>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

static void BM_SingleAlignment(benchmark::State& state)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    const int32_t genome_size        = state.range(0);

    // Generate random sequences
    std::minstd_rand rng(1);
    std::string genome_1 = genomeworks::genomeutils::generate_random_genome(genome_size, rng);
    std::string genome_2 = genomeworks::genomeutils::generate_random_sequence(genome_1, rng, genome_size / 30, genome_size / 30, genome_size / 30); // 3*x/30 = 10% difference

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
        GW_CU_CHECK_ERR(cudaStreamCreate(&s_));
    }

    ~CudaStream()
    {
        GW_CU_CHECK_ERR(cudaStreamDestroy(s_));
    }

    inline cudaStream_t& get()
    {
        return s_;
    }

private:
    cudaStream_t s_ = nullptr;
};

template <typename AlignerT>
typename std::enable_if<!std::is_same<AlignerT, AlignerGlobalMyersBanded>::value, std::unique_ptr<Aligner>>::type create_aligner_tmp_dispatch(int32_t genome_size, int32_t alignments_per_batch, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
{
    return std::make_unique<AlignerT>(
        genome_size,
        genome_size,
        alignments_per_batch,
        allocator,
        stream,
        device_id);
}

template <typename AlignerT>
typename std::enable_if<std::is_same<AlignerT, AlignerGlobalMyersBanded>::value, std::unique_ptr<Aligner>>::type create_aligner_tmp_dispatch(int32_t genome_size, int32_t alignments_per_batch, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
{
    const int64_t max_device_memory = -1;
    const int32_t max_bandwidth     = 1024;
    return std::make_unique<AlignerT>(
        max_device_memory,
        max_bandwidth,
        allocator,
        stream,
        device_id);
}

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
        aligner = create_aligner_tmp_dispatch<AlignerT>(
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
            std::string genome_1 = genomeworks::genomeutils::generate_random_genome(genome_size, rng);
            std::string genome_2 = genomeworks::genomeutils::generate_random_sequence(genome_1, rng, genome_size / 30, genome_size / 30, genome_size / 30); // 3*x/30 = 10% difference
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

} // namespace genomeworks

} // namespace claraparabricks

BENCHMARK_MAIN();
