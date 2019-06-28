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

#include "multi_batch.hpp"
#include "file_location.hpp"
#include "../common/utils.hpp"
#include "cudautils/cudautils.hpp"

namespace cga
{

namespace cudapoa
{

static void BM_MultiBatchTest(benchmark::State& state)
{
    // TODO: Query size per window from CUDAPOA API
    //       Assuming 9MB per window.
    const size_t size_per_window = 9 * 1024 * 1024;

    // Query free total and free GPU memory.
    size_t free, total;
    CGA_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));

    int32_t batches     = state.range(0);
    int32_t batch_size  = state.range(1);
    size_t total_memory = (batches * batch_size * size_per_window);

    if (total_memory >= free)
    {
        state.SkipWithError("Not enough available memory for config, skipping");
    }
    else if (total_memory < (free / 2))
    {
        state.SkipWithError("Config using less than half of available memory, skipping");
    }
    else
    {
        const int32_t total_windows = 5500;
        MultiBatch mb(batches, batch_size, std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt", total_windows);
        for (auto _ : state)
        {
            mb.process_batches();
        }
    }
}

static void CustomArguments(benchmark::internal::Benchmark* b)
{
    const int32_t min_total_windows = 512;
    const int32_t max_total_windows = 4096;
    for (int32_t batches = 1; batches <= 64; batches *= 2)
    {
        for (int32_t batch_size = 64; batch_size <= 1024; batch_size *= 2)
        {
            if (batches * batch_size <= max_total_windows && batches * batch_size >= min_total_windows)
            {
                b->Args({batches, batch_size});
            }
        }
    }
}

// Register the function as a benchmark
BENCHMARK(BM_MultiBatchTest)
    ->Unit(benchmark::kMillisecond)
    ->Apply(CustomArguments);
} // namespace cudapoa
} // namespace cga

BENCHMARK_MAIN();
