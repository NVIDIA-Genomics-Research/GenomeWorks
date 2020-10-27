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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include "../src/chainer_utils.cuh"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

TEST(TestChainerUtils, Create_Simple_Overlap_Tests)
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

    Overlap p_ab = chainerutils::create_simple_overlap(a, b, 12);
    ASSERT_EQ(p_ab.query_read_id_, static_cast<uint32_t>(12));
    ASSERT_EQ(p_ab.target_read_id_, 46);
    ASSERT_EQ(p_ab.query_start_position_in_read_, 130);
    ASSERT_EQ(p_ab.query_end_position_in_read_, 10000);
    ASSERT_EQ(p_ab.target_start_position_in_read_, 10320);
    ASSERT_EQ(p_ab.num_residues_, 12);
    ASSERT_EQ(static_cast<char>(p_ab.relative_strand), static_cast<char>(RelativeStrand::Forward));

    Anchor c;
    c.query_read_id_           = 12;
    c.query_position_in_read_  = 15000;
    c.target_read_id_          = 46;
    c.target_position_in_read_ = 16000;

    Overlap p_bc = chainerutils::create_simple_overlap(b, c, 22);
    ASSERT_EQ(p_bc.target_start_position_in_read_, 16000);
    ASSERT_EQ(p_bc.target_end_position_in_read_, 20400);
    ASSERT_EQ(static_cast<char>(p_bc.relative_strand), static_cast<char>(RelativeStrand::Reverse));
}

TEST(TestChainerUtils, Anchor_Chain_Extraction_Tests)
{

    DefaultDeviceAllocator allocator = create_default_device_allocator(2048);

    CudaStream cuda_stream = make_cuda_stream();
    auto cu_ptr            = cuda_stream.get();

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

    Overlap p_ab = chainerutils::create_simple_overlap(a, b, 2);

    Anchor c;
    c.query_read_id_           = 12;
    c.query_position_in_read_  = 15000;
    c.target_read_id_          = 46;
    c.target_position_in_read_ = 16000;

    Overlap p_bc = chainerutils::create_simple_overlap(b, c, 2);

    std::vector<Anchor> anchors;
    anchors.push_back(a);
    anchors.push_back(b);
    anchors.push_back(c);

    std::vector<Overlap> overlaps;
    overlaps.push_back(p_ab);
    overlaps.push_back(p_bc);

    device_buffer<Anchor> d_anchors(anchors.size(), allocator, cuda_stream.get());
    device_buffer<Overlap> d_overlaps(overlaps.size(), allocator, cuda_stream.get());
    cudautils::device_copy_n(anchors.data(), anchors.size(), d_anchors.data());
    cudautils::device_copy_n(overlaps.data(), overlaps.size(), d_overlaps.data());

    device_buffer<int32_t> unrolled_anchor_chains;
    device_buffer<int32_t> chain_starts;

    int32_t num_total_anchors;
    chainerutils::allocate_anchor_chains(d_overlaps,
                                         unrolled_anchor_chains,
                                         chain_starts,
                                         overlaps.size(),
                                         num_total_anchors,
                                         allocator,
                                         cu_ptr);
}

} // namespace cudamapper
} // namespace genomeworks
} // namespace claraparabricks