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

#include "../src/overlapper_triggered.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

TEST(TestCudamapperOverlapperTriggerred, OneAchorNoOverlaps)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    cudaStream_t cuda_stream;
    GW_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));
    OverlapperTriggered overlapper(allocator, cuda_stream);

    std::vector<Overlap> unfused_overlaps;
    std::vector<Anchor> anchors;

    std::vector<std::string> testv;
    testv.push_back("READ0");
    testv.push_back("READ1");
    testv.push_back("READ2");
    std::vector<std::uint32_t> test_read_length(testv.size(), 1000);

    Anchor anchor1;
    anchors.push_back(anchor1);

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, cuda_stream);
    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), cuda_stream); //H2D

    std::vector<Overlap> overlaps;
    overlapper.get_overlaps(overlaps, anchors_d, 0, 0);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    ASSERT_EQ(overlaps.size(), 0u);

    anchors_d.free();
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    GW_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperOverlapperTriggerred, FourAnchorsOneOverlap)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    cudaStream_t cuda_stream;
    GW_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));
    OverlapperTriggered overlapper(allocator, cuda_stream);

    std::vector<Overlap> unfused_overlaps;
    std::vector<Anchor> anchors;

    Anchor anchor1;
    anchor1.query_read_id_           = 1;
    anchor1.target_read_id_          = 2;
    anchor1.query_position_in_read_  = 100;
    anchor1.target_position_in_read_ = 1000;

    Anchor anchor2;
    anchor2.query_read_id_           = 1;
    anchor2.target_read_id_          = 2;
    anchor2.query_position_in_read_  = 200;
    anchor2.target_position_in_read_ = 1100;

    Anchor anchor3;
    anchor3.query_read_id_           = 1;
    anchor3.target_read_id_          = 2;
    anchor3.query_position_in_read_  = 300;
    anchor3.target_position_in_read_ = 1200;

    Anchor anchor4;
    anchor4.query_read_id_           = 1;
    anchor4.target_read_id_          = 2;
    anchor4.query_position_in_read_  = 400;
    anchor4.target_position_in_read_ = 1300;

    anchors.push_back(anchor1);
    anchors.push_back(anchor2);
    anchors.push_back(anchor3);
    anchors.push_back(anchor4);

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, cuda_stream);
    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), cuda_stream); //H2D

    std::vector<Overlap> overlaps;
    overlapper.get_overlaps(overlaps, anchors_d, false, 0, 0, 1000);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    ASSERT_EQ(overlaps.size(), 1u);
    ASSERT_EQ(overlaps[0].query_read_id_, 1u);
    ASSERT_EQ(overlaps[0].target_read_id_, 2u);
    ASSERT_EQ(overlaps[0].query_start_position_in_read_, 100u);
    ASSERT_EQ(overlaps[0].query_end_position_in_read_, 400u);
    ASSERT_EQ(overlaps[0].target_start_position_in_read_, 1000u);
    ASSERT_EQ(overlaps[0].target_end_position_in_read_, 1300u);

    anchors_d.free();
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    GW_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperOverlapperTriggerred, FourAnchorsNoOverlap)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    cudaStream_t cuda_stream;
    GW_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));
    OverlapperTriggered overlapper(allocator, cuda_stream);

    std::vector<Overlap> unfused_overlaps;
    std::vector<Anchor> anchors;

    Anchor anchor1;
    anchor1.query_read_id_           = 1;
    anchor1.target_read_id_          = 2;
    anchor1.query_position_in_read_  = 100;
    anchor1.target_position_in_read_ = 1000;

    Anchor anchor2;
    anchor2.query_read_id_           = 3;
    anchor2.target_read_id_          = 4;
    anchor2.query_position_in_read_  = 200;
    anchor2.target_position_in_read_ = 1100;

    Anchor anchor3;
    anchor3.query_read_id_           = 5;
    anchor3.target_read_id_          = 6;
    anchor3.query_position_in_read_  = 300;
    anchor3.target_position_in_read_ = 1200;

    Anchor anchor4;
    anchor4.query_read_id_           = 8;
    anchor4.target_read_id_          = 9;
    anchor4.query_position_in_read_  = 400;
    anchor4.target_position_in_read_ = 1300;

    anchors.push_back(anchor1);
    anchors.push_back(anchor2);
    anchors.push_back(anchor3);
    anchors.push_back(anchor4);

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, cuda_stream);
    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), cuda_stream); //H2D

    std::vector<Overlap> overlaps;
    overlapper.get_overlaps(overlaps, anchors_d, 0, 0, 1000);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    ASSERT_EQ(overlaps.size(), 0u);

    anchors_d.free();
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    GW_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperOverlapperTriggerred, FourColinearAnchorsOneOverlap)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    cudaStream_t cuda_stream;
    GW_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));
    OverlapperTriggered overlapper(allocator);

    std::vector<Overlap> unfused_overlaps;
    std::vector<Anchor> anchors;

    Anchor anchor1;
    anchor1.query_read_id_           = 1;
    anchor1.target_read_id_          = 2;
    anchor1.query_position_in_read_  = 100;
    anchor1.target_position_in_read_ = 1000;

    Anchor anchor2;
    anchor2.query_read_id_           = 1;
    anchor2.target_read_id_          = 2;
    anchor2.query_position_in_read_  = 200 * 10;
    anchor2.target_position_in_read_ = 1100 * 10;

    Anchor anchor3;
    anchor3.query_read_id_           = 1;
    anchor3.target_read_id_          = 2;
    anchor3.query_position_in_read_  = 300 * 10;
    anchor3.target_position_in_read_ = 1200 * 10;

    Anchor anchor4;
    anchor4.query_read_id_           = 1;
    anchor4.target_read_id_          = 2;
    anchor4.query_position_in_read_  = 400 * 10;
    anchor4.target_position_in_read_ = 1300 * 10;

    anchors.push_back(anchor1);
    anchors.push_back(anchor2);
    anchors.push_back(anchor3);
    anchors.push_back(anchor4);

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, cuda_stream);
    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), cuda_stream); //H2D

    std::vector<Overlap> overlaps;
    overlapper.get_overlaps(overlaps, anchors_d, 0, 0);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    ASSERT_EQ(overlaps.size(), 0u);

    anchors_d.free();
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    GW_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperOverlapperTriggerred, FourAnchorsLastNotInOverlap)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    cudaStream_t cuda_stream;
    GW_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));
    OverlapperTriggered overlapper(allocator);

    std::vector<Overlap> unfused_overlaps;
    std::vector<Anchor> anchors;

    Anchor anchor1;
    anchor1.query_read_id_           = 1;
    anchor1.target_read_id_          = 2;
    anchor1.query_position_in_read_  = 100;
    anchor1.target_position_in_read_ = 1000;

    Anchor anchor2;
    anchor2.query_read_id_           = 1;
    anchor2.target_read_id_          = 2;
    anchor2.query_position_in_read_  = 200;
    anchor2.target_position_in_read_ = 1100;

    Anchor anchor3;
    anchor3.query_read_id_           = 1;
    anchor3.target_read_id_          = 2;
    anchor3.query_position_in_read_  = 300;
    anchor3.target_position_in_read_ = 1200;

    Anchor anchor4;
    anchor4.query_read_id_           = 1;
    anchor4.target_read_id_          = 2;
    anchor4.query_position_in_read_  = 400 + 2000;
    anchor4.target_position_in_read_ = 1300 + 2000;

    anchors.push_back(anchor1);
    anchors.push_back(anchor2);
    anchors.push_back(anchor3);
    anchors.push_back(anchor4);

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, cuda_stream);
    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), cuda_stream); //H2D

    std::vector<Overlap> overlaps;
    overlapper.get_overlaps(overlaps, anchors_d, false, 0, 0, 1000);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    ASSERT_EQ(overlaps.size(), 1u);
    ASSERT_EQ(overlaps[0].query_read_id_, 1u);
    ASSERT_EQ(overlaps[0].target_read_id_, 2u);
    ASSERT_EQ(overlaps[0].query_start_position_in_read_, 100u);
    ASSERT_EQ(overlaps[0].query_end_position_in_read_, 300u);
    ASSERT_EQ(overlaps[0].target_start_position_in_read_, 1000u);
    ASSERT_EQ(overlaps[0].target_end_position_in_read_, 1200u);

    anchors_d.free();
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    GW_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperOverlapperTriggerred, ReverseStrand)
{
    DefaultDeviceAllocator allocator = create_default_device_allocator();
    cudaStream_t cuda_stream;
    GW_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));
    OverlapperTriggered overlapper(allocator);

    std::vector<Overlap> unfused_overlaps;
    std::vector<Anchor> anchors;

    Anchor anchor1;
    anchor1.query_read_id_           = 1;
    anchor1.target_read_id_          = 2;
    anchor1.query_position_in_read_  = 100;
    anchor1.target_position_in_read_ = 1300;

    Anchor anchor2;
    anchor2.query_read_id_           = 1;
    anchor2.target_read_id_          = 2;
    anchor2.query_position_in_read_  = 200;
    anchor2.target_position_in_read_ = 1200;

    Anchor anchor3;
    anchor3.query_read_id_           = 1;
    anchor3.target_read_id_          = 2;
    anchor3.query_position_in_read_  = 300;
    anchor3.target_position_in_read_ = 1100;

    Anchor anchor4;
    anchor4.query_read_id_           = 1;
    anchor4.target_read_id_          = 2;
    anchor4.query_position_in_read_  = 400;
    anchor4.target_position_in_read_ = 1000;

    anchors.push_back(anchor1);
    anchors.push_back(anchor2);
    anchors.push_back(anchor3);
    anchors.push_back(anchor4);

    device_buffer<Anchor> anchors_d(anchors.size(), allocator, cuda_stream);
    cudautils::device_copy_n(anchors.data(), anchors.size(), anchors_d.data(), cuda_stream); //H2D

    std::vector<Overlap> overlaps;
    overlapper.get_overlaps(overlaps, anchors_d, false, 0, 0, 1000);
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    ASSERT_EQ(overlaps.size(), 1u);
    ASSERT_GT(overlaps[0].target_end_position_in_read_, overlaps[0].target_start_position_in_read_);
    ASSERT_EQ(overlaps[0].relative_strand, RelativeStrand::Reverse);
    ASSERT_EQ(char(overlaps[0].relative_strand), '-');

    anchors_d.free();
    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    GW_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperOverlapperTriggerred, OverlapPostProcessingTwoForwardOverlapsTwoFusable)
{
    DefaultDeviceAllocator allocator;
    OverlapperTriggered overlapper(allocator);

    std::vector<Overlap> overlaps;

    Overlap overlap1;
    overlap1.relative_strand                = RelativeStrand::Forward;
    overlap1.query_read_id_                 = 20;
    overlap1.target_read_id_                = 22;
    overlap1.query_start_position_in_read_  = 1000;
    overlap1.query_end_position_in_read_    = 2000;
    overlap1.target_start_position_in_read_ = 4000;
    overlap1.target_end_position_in_read_   = 5000;
    overlaps.push_back(overlap1);

    Overlap overlap2;
    overlap2.relative_strand                = RelativeStrand::Forward;
    overlap2.query_read_id_                 = 20;
    overlap2.target_read_id_                = 22;
    overlap2.query_start_position_in_read_  = 2100;
    overlap2.query_end_position_in_read_    = 3100;
    overlap2.target_start_position_in_read_ = 5100;
    overlap2.target_end_position_in_read_   = 6100;
    overlaps.push_back(overlap2);

    Overlap overlap3;
    overlap3.relative_strand                = RelativeStrand::Forward;
    overlap3.query_read_id_                 = 55;
    overlap3.target_read_id_                = 90;
    overlap3.query_start_position_in_read_  = 1000;
    overlap3.query_end_position_in_read_    = 2000;
    overlap3.target_start_position_in_read_ = 4000;
    overlap3.target_end_position_in_read_   = 5000;
    overlaps.push_back(overlap3);

    Overlap overlap4;
    overlap4.relative_strand                = RelativeStrand::Forward;
    overlap4.query_read_id_                 = 55;
    overlap4.target_read_id_                = 90;
    overlap4.query_start_position_in_read_  = 2100;
    overlap4.query_end_position_in_read_    = 3100;
    overlap4.target_start_position_in_read_ = 5100;
    overlap4.target_end_position_in_read_   = 6100;
    overlaps.push_back(overlap4);

    Overlapper::post_process_overlaps(overlaps);

    //2 new overlaps are added
    ASSERT_EQ(overlaps.size(), 6u);
}

TEST(TestCudamapperOverlapperTriggerred, OverlapPostProcessingTwoForwardOverlapsOneFusable)
{
    DefaultDeviceAllocator allocator;
    OverlapperTriggered overlapper(allocator);

    std::vector<Overlap> overlaps;

    Overlap overlap1;
    overlap1.relative_strand                = RelativeStrand::Forward;
    overlap1.query_read_id_                 = 20;
    overlap1.target_read_id_                = 22;
    overlap1.query_start_position_in_read_  = 1000;
    overlap1.query_end_position_in_read_    = 2000;
    overlap1.target_start_position_in_read_ = 4000;
    overlap1.target_end_position_in_read_   = 5000;
    overlaps.push_back(overlap1);

    Overlap overlap2;
    overlap2.relative_strand                = RelativeStrand::Forward;
    overlap2.query_read_id_                 = 20;
    overlap2.target_read_id_                = 22;
    overlap2.query_start_position_in_read_  = 2100;
    overlap2.query_end_position_in_read_    = 3100;
    overlap2.target_start_position_in_read_ = 5100;
    overlap2.target_end_position_in_read_   = 6100;
    overlaps.push_back(overlap2);

    Overlap overlap3;
    overlap3.relative_strand                = RelativeStrand::Forward;
    overlap3.query_read_id_                 = 55;
    overlap3.target_read_id_                = 90;
    overlap3.query_start_position_in_read_  = 1000;
    overlap3.query_end_position_in_read_    = 2000;
    overlap3.target_start_position_in_read_ = 4000;
    overlap3.target_end_position_in_read_   = 5000;
    overlaps.push_back(overlap3);

    Overlap overlap4;
    overlap4.relative_strand                = RelativeStrand::Forward;
    overlap4.query_read_id_                 = 55;
    overlap4.target_read_id_                = 91;
    overlap4.query_start_position_in_read_  = 2100;
    overlap4.query_end_position_in_read_    = 3100;
    overlap4.target_start_position_in_read_ = 5100;
    overlap4.target_end_position_in_read_   = 6100;
    overlaps.push_back(overlap4);

    Overlapper::post_process_overlaps(overlaps);

    //2 new overlaps are added
    ASSERT_EQ(overlaps.size(), 5u);
}

TEST(TestCudamapperOverlapperTriggerred, OverlapPostProcessingOneForwardOneReverseBothFuasble)
{
    DefaultDeviceAllocator allocator;
    OverlapperTriggered overlapper(allocator);

    std::vector<Overlap> overlaps;

    Overlap overlap1;
    overlap1.relative_strand                = RelativeStrand::Forward;
    overlap1.query_read_id_                 = 20;
    overlap1.target_read_id_                = 22;
    overlap1.query_start_position_in_read_  = 1000;
    overlap1.query_end_position_in_read_    = 2000;
    overlap1.target_start_position_in_read_ = 4000;
    overlap1.target_end_position_in_read_   = 5000;
    overlaps.push_back(overlap1);

    Overlap overlap2;
    overlap2.relative_strand                = RelativeStrand::Forward;
    overlap2.query_read_id_                 = 20;
    overlap2.target_read_id_                = 22;
    overlap2.query_start_position_in_read_  = 2100;
    overlap2.query_end_position_in_read_    = 3100;
    overlap2.target_start_position_in_read_ = 5100;
    overlap2.target_end_position_in_read_   = 6100;
    overlaps.push_back(overlap2);

    Overlap overlap3;
    overlap3.relative_strand                = RelativeStrand::Reverse;
    overlap3.query_read_id_                 = 55;
    overlap3.target_read_id_                = 90;
    overlap3.query_start_position_in_read_  = 1000;
    overlap3.query_end_position_in_read_    = 2000;
    overlap3.target_start_position_in_read_ = 4000;
    overlap3.target_end_position_in_read_   = 5000;
    overlaps.push_back(overlap3);

    Overlap overlap4;
    overlap4.relative_strand                = RelativeStrand::Reverse;
    overlap4.query_read_id_                 = 55;
    overlap4.target_read_id_                = 90;
    overlap4.query_start_position_in_read_  = 2100;
    overlap4.query_end_position_in_read_    = 3100;
    overlap4.target_start_position_in_read_ = 2900;
    overlap4.target_end_position_in_read_   = 3900;
    overlaps.push_back(overlap4);

    Overlapper::post_process_overlaps(overlaps);

    //2 new overlaps are added
    ASSERT_EQ(overlaps.size(), 6u);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
