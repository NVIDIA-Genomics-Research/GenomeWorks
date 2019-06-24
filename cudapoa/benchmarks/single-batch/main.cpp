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

#include "single_batch.hpp"
#include "file_location.hpp"

namespace genomeworks
{

namespace cudapoa
{

static void BM_SingleBatchTest(benchmark::State& state)
{
    SingleBatch sb(state.range(0), std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt", state.range(0));
    for (auto _ : state)
    {
        state.PauseTiming();
        sb.add_windows();
        state.ResumeTiming();
        sb.process_consensus();
    }
}

// Register the function as a benchmark
BENCHMARK(BM_SingleBatchTest)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 10);
} // namespace cudapoa
} // namespace genomeworks

BENCHMARK_MAIN();
