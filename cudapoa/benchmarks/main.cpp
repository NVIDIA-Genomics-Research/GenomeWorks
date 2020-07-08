

#include "multi_batch.hpp"
#include "single_batch.hpp"
#include "file_location.hpp"

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/cudapoa/utils.hpp>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

namespace claraparabricks
{

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

static void CustomArguments(benchmark::internal::Benchmark* b)
{
    const int32_t min_total_windows = 512;
    const int32_t max_total_windows = 4096;
    for (int32_t batches = 1; batches <= 16; batches *= 2)
    {
        b->Args({batches});
    }
}

static void BM_MultiBatchTest(benchmark::State& state)
{
    int32_t batches             = state.range(0);
    const int32_t total_windows = 5500;
    MultiBatch mb(batches, std::string(CUDAPOA_BENCHMARK_DATA_DIR) + "/sample-windows.txt", total_windows);
    for (auto _ : state)
    {
        mb.process_batches();
    }
}

// Register the functions as a benchmark
BENCHMARK(BM_SingleBatchTest)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 1 << 10);
BENCHMARK(BM_MultiBatchTest)
    ->Unit(benchmark::kMillisecond)
    ->Apply(CustomArguments);
} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks

BENCHMARK_MAIN();
