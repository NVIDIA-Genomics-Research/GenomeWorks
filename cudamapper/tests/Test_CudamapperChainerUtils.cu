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

#include "gtest/gtest.h"

#include <vector>

#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include "../src/chainer_utils.cuh"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

TEST(TestChainerUtils, Create_Overlap_Tests)
{
    Anchor a;
    a.query_read_id_           = 12;
    a.query_position_in_read_  = 130;
    a.target_read_id_          = 46;
    a.target_position_in_read_ = 10320;

    Anchor b;
    b.query_read_id_           = 12;
    b.query_position_in_read_  = 10000;
    b.target_read_id_          = 46;
    b.target_position_in_read_ = 20400;

    Overlap p_ab = chainerutils::create_overlap(a, b, 12);
    ASSERT_EQ(p_ab.query_read_id_, static_cast<uint32_t>(12));
    ASSERT_EQ(p_ab.target_read_id_, 46);
    ASSERT_EQ(p_ab.query_start_position_in_read_, 130);
    ASSERT_EQ(p_ab.query_end_position_in_read_, 10000);
    ASSERT_EQ(p_ab.target_start_position_in_read_, 10320);
    ASSERT_EQ(p_ab.num_residues_, 12);
    // ASSERT_EQ(static_cast<char>(p_ab.relative_strand), static_cast<char>(RelativeStrand::Forward));

    ASSERT_EQ(p_ab.relative_strand, RelativeStrand::Forward);

    Anchor c;
    c.query_read_id_           = 12;
    c.query_position_in_read_  = 15000;
    c.target_read_id_          = 46;
    c.target_position_in_read_ = 16000;

    Overlap p_bc = chainerutils::create_overlap(b, c, 22);
    ASSERT_EQ(p_bc.target_start_position_in_read_, 16000);
    ASSERT_EQ(p_bc.target_end_position_in_read_, 20400);
    ASSERT_EQ(p_bc.relative_strand, RelativeStrand::Reverse);
}

TEST(TestChainerUtils, Anchor_Backtrace_Tests)
{
    Anchor a;
    a.query_read_id_           = 12;
    a.query_position_in_read_  = 130;
    a.target_read_id_          = 46;
    a.target_position_in_read_ = 10320;

    Anchor b;
    b.query_read_id_           = 12;
    b.query_position_in_read_  = 10000;
    b.target_read_id_          = 46;
    b.target_position_in_read_ = 20400;

    Anchor c;
    c.query_read_id_           = 12;
    c.query_position_in_read_  = 15000;
    c.target_read_id_          = 46;
    c.target_position_in_read_ = 1000;

    std::vector<Anchor> anchors       = {a, b, c};
    std::vector<float> scores         = {10, 20, 25};
    std::vector<int32_t> predecessors = {-1, 0, 0};

    DefaultDeviceAllocator allocator = create_default_device_allocator();
    CudaStream backtrace_cuda_stream = make_cuda_stream();
    auto backtrace_cu_stream_ptr     = backtrace_cuda_stream.get();

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, backtrace_cu_stream_ptr);
    device_buffer<Overlap> overlaps_d(anchors.size(), allocator, backtrace_cu_stream_ptr);
    device_buffer<float> scores_d(anchors.size(), allocator, backtrace_cu_stream_ptr);
    device_buffer<int32_t> predecessors_d(anchors.size(), allocator, backtrace_cu_stream_ptr);
    device_buffer<int32_t> overlap_terminal_anchors(anchors.size(), allocator, backtrace_cu_stream_ptr);
    device_buffer<bool> mask_d(anchors.size(), allocator, backtrace_cu_stream_ptr);

    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), backtrace_cu_stream_ptr);
    cudautils::device_copy_n(scores.data(), scores.size(), scores_d.data(), backtrace_cu_stream_ptr);
    cudautils::device_copy_n(predecessors.data(), predecessors.size(), predecessors_d.data(), backtrace_cu_stream_ptr);

    chainerutils::backtrace_anchors_to_overlaps<<<64, 32, 0, backtrace_cu_stream_ptr>>>(anchors_d.data(),
                                                                                        overlaps_d.data(),
                                                                                        overlap_terminal_anchors.data(),
                                                                                        scores_d.data(),
                                                                                        mask_d.data(),
                                                                                        predecessors_d.data(),
                                                                                        anchors_d.size(),
                                                                                        20);

    std::vector<Overlap> overlaps;
    overlaps.resize(overlaps_d.size());
    cudautils::device_copy_n(overlaps_d.data(), overlaps_d.size(), overlaps.data(), backtrace_cu_stream_ptr);
    std::vector<int32_t> terminals_h;
    terminals_h.resize(overlaps.size());
    cudautils::device_copy_n(overlap_terminal_anchors.data(), overlap_terminal_anchors.size(), terminals_h.data(), backtrace_cu_stream_ptr);
    ASSERT_EQ(overlaps_d.size(), 3);

    ASSERT_EQ(overlaps[0].num_residues_, 1);
    ASSERT_EQ(overlaps[0].target_read_id_, UINT32_MAX);

    ASSERT_EQ(overlaps[1].num_residues_, 2);
    ASSERT_EQ(overlaps[1].query_read_id_, 12);
    ASSERT_EQ(overlaps[1].query_start_position_in_read_, 130);
    ASSERT_EQ(overlaps[1].target_start_position_in_read_, 10320);
    ASSERT_EQ(overlaps[1].query_end_position_in_read_, 10000);
    ASSERT_EQ(overlaps[1].target_end_position_in_read_, 20400);
    ASSERT_EQ(overlaps[1].relative_strand, RelativeStrand::Forward);

    ASSERT_EQ(overlaps[2].num_residues_, 2);
    ASSERT_EQ(overlaps[2].query_read_id_, 12);
    ASSERT_EQ(overlaps[2].query_end_position_in_read_, 15000);
    ASSERT_EQ(overlaps[2].relative_strand, RelativeStrand::Reverse);

    ASSERT_EQ(terminals_h[0], -1);
    ASSERT_EQ(terminals_h[1], 1);
    ASSERT_EQ(terminals_h[2], 2);
}

TEST(TestChainerUtils, Anchor_Chain_Allocation_Tests)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();

    CudaStream chain_alloc_cuda_stream = make_cuda_stream();
    auto chain_alloc_cu_stream_ptr     = chain_alloc_cuda_stream.get();

    Anchor a;
    a.query_read_id_           = 12;
    a.query_position_in_read_  = 130;
    a.target_read_id_          = 46;
    a.target_position_in_read_ = 10320;

    Anchor b;
    b.query_read_id_           = 12;
    b.query_position_in_read_  = 10000;
    b.target_read_id_          = 46;
    b.target_position_in_read_ = 20400;

    Overlap p_ab = chainerutils::create_overlap(a, b, 2);

    Anchor c;
    c.query_read_id_           = 12;
    c.query_position_in_read_  = 15000;
    c.target_read_id_          = 46;
    c.target_position_in_read_ = 16000;

    Overlap p_bc = chainerutils::create_overlap(b, c, 2);

    std::vector<Anchor> anchors = {a, b, c};

    std::vector<Overlap> overlaps = {p_ab, p_bc};

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, chain_alloc_cu_stream_ptr);
    device_buffer<Overlap> overlaps_d(overlaps.size(), allocator, chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(overlaps.data(), overlaps.size(), overlaps_d.data(), chain_alloc_cu_stream_ptr);

    device_buffer<int32_t> unrolled_anchor_chains(0, allocator, chain_alloc_cu_stream_ptr);
    device_buffer<int32_t> chain_starts(0, allocator, chain_alloc_cu_stream_ptr);
    int64_t num_total_anchors;

    chainerutils::allocate_anchor_chains(overlaps_d,
                                         unrolled_anchor_chains,
                                         chain_starts,
                                         num_total_anchors,
                                         allocator,
                                         chain_alloc_cu_stream_ptr);

    ASSERT_EQ(num_total_anchors, 4);

    std::vector<int32_t> anchors_chain_starts_h(overlaps.size());
    cudautils::device_copy_n(chain_starts.data(), num_total_anchors, anchors_chain_starts_h.data(), chain_alloc_cu_stream_ptr);
    ASSERT_EQ(anchors_chain_starts_h[0], 0);
    ASSERT_EQ(anchors_chain_starts_h[1], 2);
    ASSERT_EQ(num_total_anchors, 4);
}

TEST(TestChainerUtils, Test_Output_Overlap_Chains_By_RLE)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();

    CudaStream chain_alloc_cuda_stream = make_cuda_stream();
    auto chain_alloc_cu_stream_ptr     = chain_alloc_cuda_stream.get();

    Anchor a;
    a.query_read_id_           = 12;
    a.query_position_in_read_  = 130;
    a.target_read_id_          = 46;
    a.target_position_in_read_ = 10320;

    Anchor b;
    b.query_read_id_           = 12;
    b.query_position_in_read_  = 10000;
    b.target_read_id_          = 46;
    b.target_position_in_read_ = 20400;

    Anchor c;
    c.query_read_id_           = 12;
    c.query_position_in_read_  = 15000;
    c.target_read_id_          = 46;
    c.target_position_in_read_ = 1000;

    Anchor d;
    d.query_read_id_           = 12;
    d.query_position_in_read_  = 20000;
    d.target_read_id_          = 46;
    d.target_position_in_read_ = 6000;

    std::vector<Anchor> anchors = {a, b, c, d};

    std::vector<int32_t> run_starts_h  = {0, 2};
    std::vector<int32_t> run_lengths_h = {2, 2};
    device_buffer<int32_t> run_starts_d(run_starts_h.size(), allocator, chain_alloc_cu_stream_ptr);
    device_buffer<int32_t> run_lengths_d(run_starts_h.size(), allocator, chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(run_starts_h.data(), run_starts_h.size(), run_starts_d.data(), chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(run_lengths_h.data(), run_lengths_h.size(), run_lengths_d.data(), chain_alloc_cu_stream_ptr);

    Overlap p_ab                  = chainerutils::create_overlap(a, b, 2);
    Overlap p_cd                  = chainerutils::create_overlap(c, d, 2);
    std::vector<Overlap> overlaps = {p_ab, p_cd};
    device_buffer<Overlap> overlaps_d(overlaps.size(), allocator, chain_alloc_cu_stream_ptr);
    device_buffer<Anchor> anchors_d(anchors.size(), allocator, chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(overlaps.data(), overlaps.size(), overlaps_d.data(), chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), chain_alloc_cu_stream_ptr);

    device_buffer<int32_t> chain_starts(0, allocator, chain_alloc_cu_stream_ptr);
    device_buffer<int32_t> unrolled_anchor_chains(0, allocator, chain_alloc_cu_stream_ptr);
    int64_t num_total_anchors = 0;
    chainerutils::allocate_anchor_chains(overlaps_d,
                                         unrolled_anchor_chains,
                                         chain_starts,
                                         num_total_anchors,
                                         allocator,
                                         chain_alloc_cu_stream_ptr);

    chainerutils::output_overlap_chains_by_RLE<<<1024, 64, 0, chain_alloc_cu_stream_ptr>>>(overlaps_d.data(),
                                                                                           anchors_d.data(),
                                                                                           run_starts_d.data(),
                                                                                           run_lengths_d.data(),
                                                                                           unrolled_anchor_chains.data(),
                                                                                           chain_starts.data(),
                                                                                           overlaps.size());
    std::vector<int32_t> chain_starts_h;
    chain_starts_h.resize(chain_starts.size());
    cudautils::device_copy_n(chain_starts.data(), chain_starts.size(), chain_starts_h.data(), chain_alloc_cu_stream_ptr);
    std::vector<int32_t> chains_h;
    chains_h.resize(unrolled_anchor_chains.size());
    cudautils::device_copy_n(unrolled_anchor_chains.data(), unrolled_anchor_chains.size(), chains_h.data(), chain_alloc_cu_stream_ptr);
    ASSERT_EQ(unrolled_anchor_chains.size(), 4);
    ASSERT_EQ(chain_starts.size(), 2);
    ASSERT_EQ(chain_starts_h[0], 0);
    ASSERT_EQ(chain_starts_h[1], 2);
    ASSERT_EQ(chains_h[0], 0);
    ASSERT_EQ(chains_h[2], 2);
}

TEST(TestChainerUtils, Test_Output_Overlap_Chains_By_Backtrace)
{

    DefaultDeviceAllocator allocator = create_default_device_allocator();

    CudaStream chain_alloc_cuda_stream = make_cuda_stream();
    auto chain_alloc_cu_stream_ptr     = chain_alloc_cuda_stream.get();

    Anchor a;
    a.query_read_id_           = 12;
    a.query_position_in_read_  = 130;
    a.target_read_id_          = 46;
    a.target_position_in_read_ = 10320;

    Anchor b;
    b.query_read_id_           = 12;
    b.query_position_in_read_  = 10000;
    b.target_read_id_          = 46;
    b.target_position_in_read_ = 20400;

    Anchor c;
    c.query_read_id_           = 12;
    c.query_position_in_read_  = 15000;
    c.target_read_id_          = 46;
    c.target_position_in_read_ = 1000;

    std::vector<Anchor> anchors                   = {a, b, c};
    std::vector<float> scores                     = {10, 20, 25};
    std::vector<int32_t> predecessors             = {-1, 0, 0};
    std::vector<int32_t> overlap_terminal_anchors = {1, 2};

    Overlap p_ab                  = chainerutils::create_overlap(a, b, 2);
    Overlap p_ac                  = chainerutils::create_overlap(a, c, 2);
    std::vector<Overlap> overlaps = {p_ab, p_ac};

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, chain_alloc_cu_stream_ptr);
    device_buffer<Overlap> overlaps_d(overlaps.size(), allocator, chain_alloc_cu_stream_ptr);
    device_buffer<bool> mask_d(anchors.size(), allocator, chain_alloc_cu_stream_ptr);
    device_buffer<int32_t> pred_d(predecessors.size(), allocator, chain_alloc_cu_stream_ptr);
    device_buffer<int32_t> terminals_d(overlaps.size(), allocator, chain_alloc_cu_stream_ptr);

    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(overlaps.data(), overlaps.size(), overlaps_d.data(), chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(predecessors.data(), predecessors.size(), pred_d.data(), chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(overlap_terminal_anchors.data(), overlap_terminal_anchors.size(), terminals_d.data(), chain_alloc_cu_stream_ptr);

    device_buffer<int32_t> unrolled_anchor_chains(0, allocator, chain_alloc_cu_stream_ptr);
    device_buffer<int32_t> chain_starts(0, allocator, chain_alloc_cu_stream_ptr);
    int64_t num_total_anchors;

    chainerutils::allocate_anchor_chains(overlaps_d,
                                         unrolled_anchor_chains,
                                         chain_starts,
                                         num_total_anchors,
                                         allocator,
                                         chain_alloc_cu_stream_ptr);

    chainerutils::output_overlap_chains_by_backtrace<<<1024, 64, 0, chain_alloc_cu_stream_ptr>>>(overlaps_d.data(),
                                                                                                 anchors_d.data(),
                                                                                                 mask_d.data(),
                                                                                                 pred_d.data(),
                                                                                                 terminals_d.data(),
                                                                                                 unrolled_anchor_chains.data(),
                                                                                                 chain_starts.data(),
                                                                                                 overlaps.size(),
                                                                                                 false);

    std::vector<int32_t> unrolled_chains_h;
    unrolled_chains_h.resize(unrolled_anchor_chains.size());
    std::vector<int32_t> chain_starts_h;
    chain_starts_h.resize(chain_starts.size());
    cudautils::device_copy_n(unrolled_anchor_chains.data(), unrolled_anchor_chains.size(), unrolled_chains_h.data(), chain_alloc_cu_stream_ptr);
    cudautils::device_copy_n(chain_starts.data(), chain_starts.size(), chain_starts_h.data(), chain_alloc_cu_stream_ptr);

    ASSERT_EQ(chain_starts_h.size(), 2);
    ASSERT_EQ(chain_starts_h[0], 0);
    ASSERT_EQ(chain_starts_h[1], 2);
    ASSERT_EQ(unrolled_chains_h[0], 0);
    ASSERT_EQ(unrolled_chains_h[1], 1);
    ASSERT_EQ(unrolled_chains_h[2], 0);
    ASSERT_EQ(unrolled_chains_h[3], 2);
    ASSERT_EQ(unrolled_anchor_chains.size(), 4);
}

} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks