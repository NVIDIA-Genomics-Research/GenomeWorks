/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"
#include "../src/needleman_wunsch_cpu.hpp"
#include "../src/ukkonen_cpu.hpp"
#include "../src/ukkonen_gpu.cuh"
#include "../src/device_storage.cuh"
#include "../src/batched_device_matrices.cuh"
#include "common.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>

namespace genomeworks
{

namespace cudaaligner
{

using ::testing::TestWithParam;
using ::testing::ValuesIn;

typedef struct
{
    std::string target;
    std::string query;
    int32_t p;
} TestAlignmentPair;

std::vector<TestAlignmentPair> getTestCases()
{
    std::vector<TestAlignmentPair> test_cases;
    TestAlignmentPair t;

    // Test 1
    t.target = "ACTG";
    t.query  = "ACTG";
    t.p      = 0;
    test_cases.push_back(t);

    // Test 2
    t.target = "ACTG";
    t.query  = "ATCG";
    t.p      = 3;
    test_cases.push_back(t);

    // Test 3
    t.target = "ACTG";
    t.query  = "ATG";
    t.p      = 2;
    test_cases.push_back(t);

    // Test 4
    t.target = "ACTG";
    t.query  = "";
    t.p      = 0;
    test_cases.push_back(t);

    // Test 5
    t.target = "ACTGGTCA";
    t.query  = "ACTG";
    t.p      = 4;
    test_cases.push_back(t);

    // Test 6
    t.target = "ACTG";
    t.query  = "BDEF";
    t.p      = 4;
    test_cases.push_back(t);

    // Test 7
    t.target = generate_random_genome(5000);
    t.query  = generate_random_genome(4800);
    t.p      = 5000;
    test_cases.push_back(t);

    return test_cases;
}

class AlignerImplementation : public TestWithParam<TestAlignmentPair>
{
public:
    void SetUp()
    {
        param_ = GetParam();
    }
    void TearDown() {}

    void compare_banded_score_matrix(const matrix<int>& regular_matrix_ref, const matrix<int>& banded_matrix, int32_t p)
    {
        int32_t const m = regular_matrix_ref.num_rows();
        int32_t const n = regular_matrix_ref.num_cols();

        for (int32_t i = 0; i < m; ++i)
        {
            for (int32_t j = 0; j < n; ++j)
            {
                if (j - i >= -p && j - i <= std::abs(n - m) + p)
                {
                    int32_t k, l;
                    std::tie(k, l) = nw_cpu::to_band_indices(i, j, p);
                    ASSERT_EQ(banded_matrix(k, l), regular_matrix_ref(i, j)) << "(" << k << "," << l << ")d=(" << i << "," << j << ") -- " << banded_matrix(k, l) << " != " << regular_matrix_ref(i, j);
                }
            }
        }
    }

    void compare_backtrace(const std::vector<int8_t>& a, const std::vector<int8_t>& b)
    {
        ASSERT_EQ(a.size(), b.size()) << "Backtraces are of varying length\n"
                                      << print_backtrace(a) << "\n"
                                      << print_backtrace(b) << "\n";

        for (uint32_t i = 0; i < a.size(); i++)
        {
            ASSERT_EQ(a[i], b[i]);
        }
    }

    std::string print_backtrace(const std::vector<int8_t>& bt)
    {
        std::string out = "";
        for (auto& s : bt)
        {
            out += std::to_string(static_cast<int32_t>(s)) + " ";
        }
        return out;
    }

protected:
    TestAlignmentPair param_;
};

matrix<int> ukkonen_gpu_build_score_matrix(const std::string& target, const std::string& query, int32_t ukkonen_p)
{
    // Allocate buffers and prepare data
    int32_t query_length          = query.length();
    int32_t target_length         = target.length();
    int32_t max_path_length       = query_length + target_length;
    int32_t max_alignment_length  = std::max(query_length, target_length);
    int32_t max_length_difference = std::abs(target_length - query_length);

    auto score_matrices = std::make_unique<batched_device_matrices<nw_score_t>>(
        1, ukkonen_max_score_matrix_size(query_length, target_length, max_length_difference, ukkonen_p), nullptr, 0);

    device_storage<int8_t> path_d(max_path_length, 0);
    std::vector<int8_t> path_h(max_path_length);

    device_storage<int32_t> path_length_d(1, 0);
    std::vector<int32_t> path_length_h(1);

    device_storage<char> sequences_d(2 * max_alignment_length, 0);
    cudaMemcpy(sequences_d.data(), query.c_str(), sizeof(char) * query_length, cudaMemcpyHostToDevice);
    cudaMemcpy(sequences_d.data() + max_alignment_length, target.c_str(), sizeof(char) * target_length, cudaMemcpyHostToDevice);

    device_storage<int32_t> sequence_lengths_d(2, 0);
    cudaMemcpy(sequence_lengths_d.data(), &query_length, sizeof(int32_t) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(sequence_lengths_d.data() + 1, &target_length, sizeof(int32_t) * 1, cudaMemcpyHostToDevice);

    ukkonen_compute_score_matrix_gpu(*score_matrices.get(),
                                     sequences_d.data(), sequence_lengths_d.data(),
                                     max_length_difference, max_alignment_length, 1,
                                     ukkonen_p, nullptr);

    int32_t const m        = query_length + 1;
    int32_t const n        = target_length + 1;
    int32_t const bw       = (1 + n - m + 2 * ukkonen_p + 1) / 2;
    matrix<nw_score_t> mat = score_matrices->get_matrix(0, bw, n + m, nullptr);
    matrix<int> mat_int(bw, n + m);
    for (int j = 0; j < n + m; ++j)
        for (int i = 0; i < bw; ++i)
        {
            mat_int(i, j) = mat(i, j);
        }
    return mat_int;
}

TEST_P(AlignerImplementation, UkkonenVsNaiveScoringMatrix)
{
    matrix<int> u = nw_cpu::ukkonen_build_score_matrix(param_.target, param_.query, param_.p);
    matrix<int> r = nw_cpu::needleman_wunsch_build_score_matrix_naive(param_.target, param_.query);

    compare_banded_score_matrix(r, u, param_.p);
}

TEST_P(AlignerImplementation, UkkonenGpuVsUkkonenCpuScoringMatrix)
{
    matrix<int> u = ukkonen_gpu_build_score_matrix(param_.target, param_.query, param_.p);
    matrix<int> r = nw_cpu::ukkonen_build_score_matrix(param_.target, param_.query, param_.p);
    int const m   = param_.query.length() + 1;
    int const n   = param_.target.length() + 1;
    int const p   = param_.p;

    int32_t const bw = (1 + n - m + 2 * p + 1) / 2;
    ASSERT_EQ(u.num_rows(), bw);
    ASSERT_EQ(u.num_cols(), n + m);
    ASSERT_EQ(r.num_rows(), bw);
    ASSERT_EQ(r.num_cols(), n + m);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            if (j - i >= -p && j - i <= n - m + p)
            {
                int k, l;
                std::tie(k, l) = nw_cpu::to_band_indices(i, j, p);
                if (u(k, l) != r(k, l))
                    ASSERT_EQ(u(k, l), r(k, l)) << "(" << k << "," << l << ")d[=(" << i << "," << j << ")] -- " << u(k, l) << " != " << r(k, l) << std::endl;
            }
        }
}

std::vector<int8_t> run_ukkonen_gpu(const std::string& target, const std::string& query, int32_t ukkonen_p)
{
    // Allocate buffers and prepare data
    int32_t query_length          = query.length();
    int32_t target_length         = target.length();
    int32_t max_path_length       = query_length + target_length;
    int32_t max_alignment_length  = std::max(query_length, target_length);
    int32_t max_length_difference = std::abs(target_length - query_length);

    auto score_matrices = std::make_unique<batched_device_matrices<nw_score_t>>(
        1, ukkonen_max_score_matrix_size(query_length, target_length, max_length_difference, ukkonen_p), nullptr, 0);

    device_storage<int8_t> path_d(max_path_length, 0);
    std::vector<int8_t> path_h(max_path_length);

    device_storage<int32_t> path_length_d(1, 0);
    std::vector<int32_t> path_length_h(1);

    device_storage<char> sequences_d(2 * max_alignment_length, 0);
    cudaMemcpy(sequences_d.data(), query.c_str(), sizeof(char) * query_length, cudaMemcpyHostToDevice);
    cudaMemcpy(sequences_d.data() + max_alignment_length, target.c_str(), sizeof(char) * target_length, cudaMemcpyHostToDevice);

    device_storage<int32_t> sequence_lengths_d(2, 0);
    cudaMemcpy(sequence_lengths_d.data(), &query_length, sizeof(int32_t) * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(sequence_lengths_d.data() + 1, &target_length, sizeof(int32_t) * 1, cudaMemcpyHostToDevice);

    // Run kernel
    ukkonen_gpu(path_d.data(), path_length_d.data(), max_path_length,
                sequences_d.data(), sequence_lengths_d.data(),
                max_length_difference, max_alignment_length, 1,
                score_matrices.get(),
                ukkonen_p,
                nullptr);

    // Get results
    cudaMemcpy(path_h.data(), path_d.data(), sizeof(int8_t) * max_path_length, cudaMemcpyDeviceToHost);
    cudaMemcpy(path_length_h.data(), path_length_d.data(), sizeof(int32_t) * 1, cudaMemcpyDeviceToHost);

    std::vector<int8_t> bt;
    for (int32_t l = 0; l < path_length_h[0]; l++)
    {
        bt.push_back(path_h[l]);
    }

    std::reverse(bt.begin(), bt.end());

    return bt;
}

TEST_P(AlignerImplementation, UkkonenCpuFullVsUkkonenGpuFull)
{
    int32_t const p            = 1;
    std::vector<int8_t> cpu_bt = nw_cpu::ukkonen_cpu(param_.target, param_.query, p);
    std::vector<int8_t> gpu_bt = run_ukkonen_gpu(param_.target, param_.query, p);

    compare_backtrace(cpu_bt, gpu_bt);
}

INSTANTIATE_TEST_SUITE_P(TestNeedlemanWunschImplementation, AlignerImplementation, ValuesIn(getTestCases()));
}
}
