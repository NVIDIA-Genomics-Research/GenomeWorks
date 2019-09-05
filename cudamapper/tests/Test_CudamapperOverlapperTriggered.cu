/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <numeric>
#include "gtest/gtest.h"
#include "mock_index.cuh"
#include "cudamapper_file_location.hpp"
#include "../src/cudamapper_utils.hpp"
#include "../src/overlapper_triggered.hpp"


namespace claragenomics {

    TEST(TestCudamapperOverlapperTriggerred, FuseTwoOverlaps) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;

        Overlap overlap1;
        overlap1.query_read_id_ = 20;
        overlap1.target_read_id_ = 22;
        overlap1.query_start_position_in_read_ = 1000;
        overlap1.query_end_position_in_read_= 2000;
        overlap1.target_start_position_in_read_ = 4000;
        overlap1.target_end_position_in_read_ = 5000;
        unfused_overlaps.push_back(overlap1);

        Overlap overlap2;
        overlap2.query_read_id_ = 20;
        overlap2.target_read_id_ = 22;
        overlap2.query_start_position_in_read_ = 10000;
        overlap2.query_end_position_in_read_= 12000;
        overlap2.target_start_position_in_read_ = 14000;
        overlap2.target_end_position_in_read_ = 15000;

        unfused_overlaps.push_back(overlap2);
        auto fused_overlaps = fuse_overlaps(unfused_overlaps);

        ASSERT_EQ(fused_overlaps.size(), 1);
    }

    TEST(TestCudamapperOverlapperTriggerred, DoNotuseTwoOverlaps) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;

        Overlap overlap1;
        overlap1.query_read_id_ = 20;
        overlap1.target_read_id_ = 23;
        overlap1.query_start_position_in_read_ = 1000;
        overlap1.query_end_position_in_read_= 2000;
        overlap1.target_start_position_in_read_ = 4000;
        overlap1.target_end_position_in_read_ = 5000;
        unfused_overlaps.push_back(overlap1);

        Overlap overlap2;
        overlap2.query_read_id_ = 20;
        overlap2.target_read_id_ = 22;
        overlap2.query_start_position_in_read_ = 10000;
        overlap2.query_end_position_in_read_= 12000;
        overlap2.target_start_position_in_read_ = 14000;
        overlap2.target_end_position_in_read_ = 15000;

        unfused_overlaps.push_back(overlap2);
        auto fused_overlaps = fuse_overlaps(unfused_overlaps);

        ASSERT_EQ(fused_overlaps.size(), 2);
    }


    TEST(TestCudamapperOverlapperTriggerred, OneOverlap) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;

        Overlap overlap1;
        overlap1.query_read_id_ = 20;
        overlap1.target_read_id_ = 23;
        overlap1.query_start_position_in_read_ = 1000;
        overlap1.query_end_position_in_read_= 2000;
        overlap1.target_start_position_in_read_ = 4000;
        overlap1.target_end_position_in_read_ = 5000;
        unfused_overlaps.push_back(overlap1);

        auto fused_overlaps = fuse_overlaps(unfused_overlaps);

        ASSERT_EQ(fused_overlaps.size(), 1);
    }

    TEST(TestCudamapperOverlapperTriggerred, NoOverlaps) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;


        auto fused_overlaps = fuse_overlaps(unfused_overlaps);

        ASSERT_EQ(fused_overlaps.size(), 0);
    }

    TEST(TestCudamapperOverlapperTriggerred, Fusee3Overlapsto2) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;

        Overlap overlap1;
        overlap1.query_read_id_ = 20;
        overlap1.target_read_id_ = 23;
        overlap1.query_start_position_in_read_ = 1000;
        overlap1.query_end_position_in_read_= 2000;
        overlap1.target_start_position_in_read_ = 4000;
        overlap1.target_end_position_in_read_ = 5000;
        unfused_overlaps.push_back(overlap1);

        Overlap overlap2;
        overlap2.query_read_id_ = 20;
        overlap2.target_read_id_ = 23;
        overlap2.query_start_position_in_read_ = 10000;
        overlap2.query_end_position_in_read_= 12000;
        overlap2.target_start_position_in_read_ = 14000;
        overlap2.target_end_position_in_read_ = 15000;
        unfused_overlaps.push_back(overlap2);

        Overlap overlap3;
        overlap3.query_read_id_ = 27;
        overlap3.target_read_id_ = 29;
        overlap3.query_start_position_in_read_ = 10000;
        overlap3.query_end_position_in_read_= 12000;
        overlap3.target_start_position_in_read_ = 14000;
        overlap3.target_end_position_in_read_ = 15000;

        unfused_overlaps.push_back(overlap3);

        auto fused_overlaps = fuse_overlaps(unfused_overlaps);

        ASSERT_EQ(fused_overlaps.size(), 2);
    }

    TEST(TestCudamapperOverlapperTriggerred, OneAchorNoOverlaps) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;
        std::vector<Anchor> anchors;

        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        std::vector<std::uint32_t> test_read_length(testv.size(), 1000);

        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));
        EXPECT_CALL(test_index, read_id_to_read_length)
                .WillRepeatedly(testing::ReturnRef(test_read_length));

        Anchor anchor1;

        anchors.push_back(anchor1);

        auto overlaps = overlapper.get_overlaps(anchors, test_index);
        ASSERT_EQ(overlaps.size(), 0);
    }

    TEST(TestCudamapperOverlapperTriggerred, FourAnchorsOneOverlap) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;
        std::vector<Anchor> anchors;

        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        std::vector<std::uint32_t> test_read_length(testv.size(), 1000);

        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));
        EXPECT_CALL(test_index, read_id_to_read_length)
                .WillRepeatedly(testing::ReturnRef(test_read_length));

        Anchor anchor1;
        anchor1.query_read_id_ = 1;
        anchor1.target_read_id_ = 2;
        anchor1.query_position_in_read_ = 100;
        anchor1.target_position_in_read_ = 1000;

        Anchor anchor2;
        anchor2.query_read_id_ = 1;
        anchor2.target_read_id_ = 2;
        anchor2.query_position_in_read_ = 200;
        anchor2.target_position_in_read_ = 1100;

        Anchor anchor3;
        anchor3.query_read_id_ = 1;
        anchor3.target_read_id_ = 2;
        anchor3.query_position_in_read_ = 300;
        anchor3.target_position_in_read_ = 1200;

        Anchor anchor4;
        anchor4.query_read_id_ = 1;
        anchor4.target_read_id_ = 2;
        anchor4.query_position_in_read_ = 400;
        anchor4.target_position_in_read_ = 1300;

        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        anchors.push_back(anchor3);
        anchors.push_back(anchor4);

        auto overlaps = overlapper.get_overlaps(anchors, test_index);
        ASSERT_EQ(overlaps.size(), 1);
        ASSERT_EQ(overlaps[0].query_read_id_, 1);
        ASSERT_EQ(overlaps[0].target_read_id_, 2);
        ASSERT_EQ(overlaps[0].query_start_position_in_read_, 100);
        ASSERT_EQ(overlaps[0].query_end_position_in_read_, 400);
        ASSERT_EQ(overlaps[0].target_start_position_in_read_, 1000);
        ASSERT_EQ(overlaps[0].target_end_position_in_read_, 1300);
    }

    TEST(TestCudamapperOverlapperTriggerred, FourAnchorsNoOverlap) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;
        std::vector<Anchor> anchors;

        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        std::vector<std::uint32_t> test_read_length(testv.size(), 1000);

        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));
        EXPECT_CALL(test_index, read_id_to_read_length)
                .WillRepeatedly(testing::ReturnRef(test_read_length));

        Anchor anchor1;
        anchor1.query_read_id_ = 1;
        anchor1.target_read_id_ = 2;
        anchor1.query_position_in_read_ = 100;
        anchor1.target_position_in_read_ = 1000;

        Anchor anchor2;
        anchor2.query_read_id_ = 3;
        anchor2.target_read_id_ = 4;
        anchor2.query_position_in_read_ = 200;
        anchor2.target_position_in_read_ = 1100;

        Anchor anchor3;
        anchor3.query_read_id_ = 5;
        anchor3.target_read_id_ = 6;
        anchor3.query_position_in_read_ = 300;
        anchor3.target_position_in_read_ = 1200;

        Anchor anchor4;
        anchor4.query_read_id_ = 8;
        anchor4.target_read_id_ = 9;
        anchor4.query_position_in_read_ = 400;
        anchor4.target_position_in_read_ = 1300;

        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        anchors.push_back(anchor3);
        anchors.push_back(anchor4);

        auto overlaps = overlapper.get_overlaps(anchors, test_index);
        ASSERT_EQ(overlaps.size(), 0);
    }

    TEST(TestCudamapperOverlapperTriggerred, FourColinearAnchorsOneOverlap) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;
        std::vector<Anchor> anchors;

        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        std::vector<std::uint32_t> test_read_length(testv.size(), 1000);

        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));
        EXPECT_CALL(test_index, read_id_to_read_length)
                .WillRepeatedly(testing::ReturnRef(test_read_length));

        Anchor anchor1;
        anchor1.query_read_id_ = 1;
        anchor1.target_read_id_ = 2;
        anchor1.query_position_in_read_ = 100;
        anchor1.target_position_in_read_ = 1000;

        Anchor anchor2;
        anchor2.query_read_id_ = 1;
        anchor2.target_read_id_ = 2;
        anchor2.query_position_in_read_ = 200 * 10;
        anchor2.target_position_in_read_ = 1100 * 10;

        Anchor anchor3;
        anchor3.query_read_id_ = 1;
        anchor3.target_read_id_ = 2;
        anchor3.query_position_in_read_ = 300 * 10;
        anchor3.target_position_in_read_ = 1200 * 10;

        Anchor anchor4;
        anchor4.query_read_id_ = 1;
        anchor4.target_read_id_ = 2;
        anchor4.query_position_in_read_ = 400 * 10;
        anchor4.target_position_in_read_ = 1300 * 10;

        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        anchors.push_back(anchor3);
        anchors.push_back(anchor4);

        auto overlaps = overlapper.get_overlaps(anchors, test_index);
        ASSERT_EQ(overlaps.size(), 0);
    }

    TEST(TestCudamapperOverlapperTriggerred, FourAnchorsLastNotInOverlap) {
        OverlapperTriggered overlapper;

        std::vector<Overlap> unfused_overlaps;
        std::vector<Anchor> anchors;

        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        std::vector<std::uint32_t> test_read_length(testv.size(), 1000);

        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));
        EXPECT_CALL(test_index, read_id_to_read_length)
                .WillRepeatedly(testing::ReturnRef(test_read_length));

        Anchor anchor1;
        anchor1.query_read_id_ = 1;
        anchor1.target_read_id_ = 2;
        anchor1.query_position_in_read_ = 100;
        anchor1.target_position_in_read_ = 1000;

        Anchor anchor2;
        anchor2.query_read_id_ = 1;
        anchor2.target_read_id_ = 2;
        anchor2.query_position_in_read_ = 200;
        anchor2.target_position_in_read_ = 1100;

        Anchor anchor3;
        anchor3.query_read_id_ = 1;
        anchor3.target_read_id_ = 2;
        anchor3.query_position_in_read_ = 300;
        anchor3.target_position_in_read_ = 1200;

        Anchor anchor4;
        anchor4.query_read_id_ = 1;
        anchor4.target_read_id_ = 2;
        anchor4.query_position_in_read_ = 400 + 2000;
        anchor4.target_position_in_read_ = 1300 + 2000;

        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        anchors.push_back(anchor3);
        anchors.push_back(anchor4);

        auto overlaps = overlapper.get_overlaps(anchors, test_index);
        ASSERT_EQ(overlaps.size(), 1);
        ASSERT_EQ(overlaps[0].query_read_id_, 1);
        ASSERT_EQ(overlaps[0].target_read_id_, 2);
        ASSERT_EQ(overlaps[0].query_start_position_in_read_, 100);
        ASSERT_EQ(overlaps[0].query_end_position_in_read_, 300);
        ASSERT_EQ(overlaps[0].target_start_position_in_read_, 1000);
        ASSERT_EQ(overlaps[0].target_end_position_in_read_, 1200);
    }
}
