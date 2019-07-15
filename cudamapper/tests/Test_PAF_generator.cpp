
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
#include "../src/mock_index.hpp"

namespace claragenomics {

    TEST(TestPAFGenerator, TestGenerateNoOverlapForOneAnchor){
        Matcher::Anchor anchor{0,1,2,3};
        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor);

        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");

        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));

        std::vector<Overlap> overlaps = get_overlaps(anchors, test_index);
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

        anchor1.query_read_id_ = 0;
        anchor1.target_read_id_ = 1;

        anchor2.query_read_id_ = 2;
        anchor2.target_read_id_ = 3;

        //Mock the index
        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        testv.push_back("READ3");
        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);

        std::vector<Overlap> overlaps = get_overlaps(anchors, test_index);
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

        anchor1.query_read_id_ = 0;
        anchor1.target_read_id_ = 1;

        anchor2.query_read_id_ = 0;
        anchor2.target_read_id_ = 1;

        //Mock the index
        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        testv.push_back("READ3");

        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);

        std::vector<Overlap> overlaps = get_overlaps(anchors, test_index);
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

        anchor1.query_read_id_ = 0;
        anchor1.target_read_id_ = 1;

        anchor2.query_read_id_ = 2;
        anchor2.target_read_id_ = 3;

        anchor3.query_read_id_ = 0;
        anchor3.target_read_id_ = 1;

        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        testv.push_back("READ3");
        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        anchors.push_back(anchor3);

        std::vector<Overlap> overlaps = get_overlaps(anchors, test_index);
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

        anchor1.query_read_id_ = 0;
        anchor1.target_read_id_ = 1;

        anchor2.query_read_id_ = 0;
        anchor2.target_read_id_ = 1;

        anchor3.query_read_id_ = 2;
        anchor3.target_read_id_ = 3;

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);
        anchors.push_back(anchor3);

        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        testv.push_back("READ3");
        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));

        std::vector<Overlap> overlaps = get_overlaps(anchors, test_index);
        int num_overlaps_found = overlaps.size();
        EXPECT_EQ(num_overlaps_found, 1);
    }

    TEST(TestPAFGenerator, TestOverlapHasCorrectQueryName){
        Matcher::Anchor anchor1;
        Matcher::Anchor anchor2;

        anchor1.query_position_in_read_ = 1000;
        anchor1.target_position_in_read_ = 1000;

        anchor2.query_position_in_read_ = 2000;
        anchor2.target_position_in_read_ = 2000;

        anchor1.query_read_id_ = 0;
        anchor1.target_read_id_ = 1;

        anchor2.query_read_id_ = 0;
        anchor2.target_read_id_ = 1;

        //Mock the index
        MockIndex test_index;
        std::vector<std::string> testv;
        testv.push_back("READ0");
        testv.push_back("READ1");
        testv.push_back("READ2");
        testv.push_back("READ3");

        EXPECT_CALL(test_index, read_id_to_read_name)
                .WillRepeatedly(testing::ReturnRef(testv));

        std::vector<Matcher::Anchor> anchors;
        anchors.push_back(anchor1);
        anchors.push_back(anchor2);

        std::vector<Overlap> overlaps = get_overlaps(anchors, test_index);
        int num_overlaps_found = overlaps.size();
        EXPECT_EQ(num_overlaps_found, 1);
        EXPECT_EQ(overlaps[0].query_read_name_, testv[0]);
        EXPECT_EQ(overlaps[0].target_read_name_, testv[1]);
    }

}
