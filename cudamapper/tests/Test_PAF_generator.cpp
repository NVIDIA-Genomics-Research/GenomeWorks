
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
#include "../src/matcher.hpp"
#include "../src/overlapper.hpp"

namespace claragenomics {

    TEST(TestPAFGenerator, TestGenerateNoOverlapForOneAnchor){
        Matcher::Anchor anchor{0,1,2,3};
        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor);
        std::vector<Overlap> overlaps = get_overlaps(anchors);
        int num_overlaps_found = overlaps.size();
        EXPECT_EQ(num_overlaps_found, 0);
    }

    TEST(TestPAFGenerator, TestGenerateNoOverlapForTwoAnchorsDifferentReadpairs){
        Matcher::Anchor anchor1;
        Matcher::Anchor anchor2;

        anchor1.query_position_in_read_ = 1000;
        anchor1.target_position_in_read_ = 1000;

        anchor2.query_position_in_read_ = 1000;
        anchor2.target_position_in_read_ = 1000;

        anchor1.query_read_id_ = 1;
        anchor1.target_read_id_ = 2;

        anchor2.query_read_id_ = 3;
        anchor2.target_read_id_ = 4;

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        std::vector<Overlap> overlaps = get_overlaps(anchors);
        int num_overlaps_found = overlaps.size();
        EXPECT_EQ(num_overlaps_found, 0);
    }

    TEST(TestPAFGenerator, TestGenerateOverlapForTwoAnchorsSameReadpairs){
        Matcher::Anchor anchor1;
        Matcher::Anchor anchor2;

        anchor1.query_position_in_read_ = 1000;
        anchor1.target_position_in_read_ = 1000;

        anchor2.query_position_in_read_ = 2000;
        anchor2.target_position_in_read_ = 2000;

        anchor1.query_read_id_ = 1;
        anchor1.target_read_id_ = 2;

        anchor2.query_read_id_ = 1;
        anchor2.target_read_id_ = 2;

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        std::vector<Overlap> overlaps = get_overlaps(anchors);
        int num_overlaps_found = overlaps.size();
        EXPECT_EQ(num_overlaps_found, 1);
    }

    TEST(TestPAFGenerator, TestGenerateOverlapForThreeAnchorsSameReadpairs){
        Matcher::Anchor anchor1;
        Matcher::Anchor anchor2;
        Matcher::Anchor anchor3;

        anchor1.query_position_in_read_ = 1000;
        anchor1.target_position_in_read_ = 1000;

        anchor2.query_position_in_read_ = 2000;
        anchor2.target_position_in_read_ = 2000;

        anchor3.query_position_in_read_ = 1500;
        anchor3.target_position_in_read_ = 1500;

        anchor1.query_read_id_ = 1;
        anchor1.target_read_id_ = 2;

        anchor2.query_read_id_ = 1;
        anchor2.target_read_id_ = 2;

        anchor3.query_read_id_ = 1;
        anchor3.target_read_id_ = 2;

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        anchors.push_back(anchor3);

        std::vector<Overlap> overlaps = get_overlaps(anchors);
        int num_overlaps_found = overlaps.size();
        EXPECT_EQ(num_overlaps_found, 1);
    }

    TEST(TestPAFGenerator, TestGenerateOverlap2){
        Matcher::Anchor anchor1;
        Matcher::Anchor anchor2;
        Matcher::Anchor anchor3;

        anchor1.query_position_in_read_ = 1000;
        anchor1.target_position_in_read_ = 1000;

        anchor2.query_position_in_read_ = 2000;
        anchor2.target_position_in_read_ = 2000;

        anchor3.query_position_in_read_ = 1500;
        anchor3.target_position_in_read_ = 1500;

        anchor1.query_read_id_ = 1;
        anchor1.target_read_id_ = 2;

        anchor2.query_read_id_ = 1;
        anchor2.target_read_id_ = 2;

        anchor3.query_read_id_ = 7;
        anchor3.target_read_id_ = 8;

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        anchors.push_back(anchor3);

        std::vector<Overlap> overlaps = get_overlaps(anchors);
        int num_overlaps_found = overlaps.size();
        print_paf(overlaps);
        EXPECT_EQ(num_overlaps_found, 1);
    }

    TEST(TestPAFGenerator, TestNumberOfOverlapsPrinted) {
        //TODO: this requires implementation. Need to capture stdout from this function.
    }
}
