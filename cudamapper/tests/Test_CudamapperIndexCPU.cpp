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
#include "cudamapper_file_location.hpp"
#include "../src/index_cpu.hpp"
#include "../src/index_generator_cpu.hpp"

namespace claragenomics {

    TEST(TestCudamapperIndexCPU, OneReadOneMinimizer) {
        // >read_0
        // GATT

        // GATT = 0x2033
        // AATC = 0x0031 <- minimizer

        IndexGeneratorCPU index_generator(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) +  "/one_read_one_minimizer.fasta", 4, 1);
        IndexCPU index(index_generator);

        ASSERT_EQ(index.number_of_reads(), 1);

        const std::vector<std::string>& read_id_to_read_name = index.read_id_to_read_name();
        ASSERT_EQ(read_id_to_read_name.size(), 1);
        EXPECT_EQ(read_id_to_read_name[0], std::string("read_0"));

        const std::vector<position_in_read_t>& positions_in_reads = index.positions_in_reads();
        ASSERT_EQ(positions_in_reads.size(), 1);
        EXPECT_EQ(positions_in_reads[0], 0);

        const std::vector<read_id_t>& read_ids = index.read_ids();
        ASSERT_EQ(read_ids.size(), 1);
        EXPECT_EQ(read_ids[0], 0);

        const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads = index.directions_of_reads();
        ASSERT_EQ(directions_of_reads.size(), 1);
        EXPECT_EQ(directions_of_reads[0], SketchElement::DirectionOfRepresentation::REVERSE);

        const std::vector<std::map<representation_t, ArrayBlock>>& read_id_and_representation_to_all_its_sketch_elements = index.read_id_and_representation_to_all_its_sketch_elements();
        ASSERT_EQ(read_id_and_representation_to_all_its_sketch_elements.size(), 1);
        ASSERT_NE(read_id_and_representation_to_all_its_sketch_elements[0].find(0x0031), read_id_and_representation_to_all_its_sketch_elements[0].end());
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x0031).first_element_, 0);
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x0031).block_size_, 1);

        const std::map<representation_t, ArrayBlock>& representation_to_all_its_sketch_elements = index.representation_to_all_its_sketch_elements();
        ASSERT_EQ(representation_to_all_its_sketch_elements.size(), 1);
        ASSERT_NE(representation_to_all_its_sketch_elements.find(0x0031), representation_to_all_its_sketch_elements.end());
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x0031).first_element_, 0);
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x0031).block_size_, 1);
    }

    TEST(TestCudamapperIndexCPU, TwoReadsMultipleMiniminizers) {
        // >read_0
        // CATCAAG
        // >read_1
        // AAGCTA

        // CATCAAG
        // Central minimizers:
        // CATC: CAT, ATG, <ATC>, GAT
        // ATCA: <ATC>, GAT, TCA, TGA
        // TCAA: TCA, TGA, <CAA>, TTG
        // CAAG: CAA, TTG, <AAG>, CTT
        // front end minimizers: CAT, <ATG>
        // beck end minimizers: none
        // All minimizers: ATC(1f), CAA(3f), AAG(4f), ATG(0r)

        // AAGCTA
        // Central minimizers:
        // AAGC: <AAG>, CTT, AGC, GCT
        // AGCT: <AGC>, GCT, GCT, <AGC>
        // GCTA: GCT, <AGC>, CTA, TAG
        // Front end minimizers: none
        // Back end miniminers: <CTA>, TAG
        // All minimizers: AAG(0f), AGC(1f), AGC(2r), CTA(3f)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0         1         2         3         4         5         6         7
        // data arrays: AAG(4f0), AAG(0f1), AGC(1f1), AGC(2r1), ATC(1f0), ATG(0r0), CA(3f0), CTA(3f1)
        // read_id_and_representation_to_all_its_sketch_elements: read_0(AAG(0,1),ATC(4,1),ATG(5,1),CAA(6,1)), read_1(AAG(1,1),AGC(2,2),CTA(7,1))
        // representation_to_all_its_sketch_elements: AAG(0,2),AGC(2,2),ATC(4,1),ATG(5,1),CAA(6,1),CTA(7,1)

        IndexGeneratorCPU index_generator(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/two_reads_multiple_minimizers.fasta", 3, 2);
        IndexCPU index(index_generator);

        ASSERT_EQ(index.number_of_reads(), 2);

        const std::vector<std::string>& read_id_to_read_name = index.read_id_to_read_name();
        ASSERT_EQ(read_id_to_read_name.size(), 2);
        EXPECT_EQ(read_id_to_read_name[0], std::string("read_0"));
        EXPECT_EQ(read_id_to_read_name[1], std::string("read_1"));

        const std::vector<position_in_read_t>& positions_in_reads = index.positions_in_reads();
        ASSERT_EQ(positions_in_reads.size(), 8);
        const std::vector<read_id_t>& read_ids = index.read_ids();
        ASSERT_EQ(read_ids.size(), 8);
        const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads = index.directions_of_reads();
        ASSERT_EQ(directions_of_reads.size(), 8);

        // Test data arrays

        // AAG(4f0)
        EXPECT_EQ(positions_in_reads[0], 4);
        EXPECT_EQ(read_ids[0], 0);
        EXPECT_EQ(directions_of_reads[0], SketchElement::DirectionOfRepresentation::FORWARD);
        // AAG(0f1)
        EXPECT_EQ(positions_in_reads[1], 0);
        EXPECT_EQ(read_ids[1], 1);
        EXPECT_EQ(directions_of_reads[1], SketchElement::DirectionOfRepresentation::FORWARD);
        // AGC(1f1)
        EXPECT_EQ(positions_in_reads[2], 1);
        EXPECT_EQ(read_ids[2], 1);
        EXPECT_EQ(directions_of_reads[2], SketchElement::DirectionOfRepresentation::FORWARD);
        // AGC(2r1)
        EXPECT_EQ(positions_in_reads[3], 2);
        EXPECT_EQ(read_ids[3], 1);
        EXPECT_EQ(directions_of_reads[3], SketchElement::DirectionOfRepresentation::REVERSE);
        // ATC(1f0)
        EXPECT_EQ(positions_in_reads[4], 1);
        EXPECT_EQ(read_ids[4], 0);
        EXPECT_EQ(directions_of_reads[4], SketchElement::DirectionOfRepresentation::FORWARD);
        // ATG(0r0)
        EXPECT_EQ(positions_in_reads[5], 0);
        EXPECT_EQ(read_ids[5], 0);
        EXPECT_EQ(directions_of_reads[5], SketchElement::DirectionOfRepresentation::REVERSE);
        // CAA(3f0)
        EXPECT_EQ(positions_in_reads[6], 3);
        EXPECT_EQ(read_ids[6], 0);
        EXPECT_EQ(directions_of_reads[6], SketchElement::DirectionOfRepresentation::FORWARD);
        // CTA(3f1)
        EXPECT_EQ(positions_in_reads[7], 3);
        EXPECT_EQ(read_ids[7], 1);
        EXPECT_EQ(directions_of_reads[7], SketchElement::DirectionOfRepresentation::FORWARD);

        // Test read_id_and_representation_to_all_its_sketch_elements
        // read_id_and_representation_to_all_its_sketch_elements: read_0(AAG(0,1),ATC(4,1),ATG(5,1),CAA(6,1)), read_1(AAG(1,1),AGC(2,2),CTA(7,1))
        const std::vector<std::map<representation_t, ArrayBlock>>& read_id_and_representation_to_all_its_sketch_elements = index.read_id_and_representation_to_all_its_sketch_elements();
        ASSERT_EQ(read_id_and_representation_to_all_its_sketch_elements.size(), 2);
        ASSERT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].size(), 4);
        ASSERT_EQ(read_id_and_representation_to_all_its_sketch_elements[1].size(), 3);
        // read_0 AAG(0,1)
        ASSERT_NE(read_id_and_representation_to_all_its_sketch_elements[0].find(0x002), read_id_and_representation_to_all_its_sketch_elements[0].end());
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x002).first_element_, 0);
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x002).block_size_, 1);
        // read_0 ATC(4,1)
        ASSERT_NE(read_id_and_representation_to_all_its_sketch_elements[0].find(0x031), read_id_and_representation_to_all_its_sketch_elements[0].end());
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x031).first_element_, 4);
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x031).block_size_, 1);
        // read_0 ATG(5,1)
        ASSERT_NE(read_id_and_representation_to_all_its_sketch_elements[0].find(0x032), read_id_and_representation_to_all_its_sketch_elements[0].end());
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x032).first_element_, 5);
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x032).block_size_, 1);
        // read_0 CAA(6,1)
        ASSERT_NE(read_id_and_representation_to_all_its_sketch_elements[0].find(0x100), read_id_and_representation_to_all_its_sketch_elements[0].end());
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x100).first_element_, 6);
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[0].at(0x100).block_size_, 1);
        // read_1 AAG(1,1)
        ASSERT_NE(read_id_and_representation_to_all_its_sketch_elements[1].find(0x002), read_id_and_representation_to_all_its_sketch_elements[1].end());
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[1].at(0x002).first_element_, 1);
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[1].at(0x002).block_size_, 1);
        // read_1 AGC(2,2)
        ASSERT_NE(read_id_and_representation_to_all_its_sketch_elements[1].find(0x021), read_id_and_representation_to_all_its_sketch_elements[1].end());
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[1].at(0x021).first_element_, 2);
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[1].at(0x021).block_size_, 2);
        // read_1 CTA(7,1)
        ASSERT_NE(read_id_and_representation_to_all_its_sketch_elements[1].find(0x130), read_id_and_representation_to_all_its_sketch_elements[1].end());
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[1].at(0x130).first_element_, 7);
        EXPECT_EQ(read_id_and_representation_to_all_its_sketch_elements[1].at(0x130).block_size_, 1);

        // Test representation_to_all_its_sketch_elements
        // representation_to_all_its_sketch_elements: AAG(0,2),AGC(2,2),ATC(4,1),ATG(5,1),CAA(6,1),CTA(7,1)
        const std::map<representation_t, ArrayBlock>& representation_to_all_its_sketch_elements = index.representation_to_all_its_sketch_elements();
        ASSERT_EQ(representation_to_all_its_sketch_elements.size(), 6);
        // AAG(0,2)
        ASSERT_NE(representation_to_all_its_sketch_elements.find(0x002), representation_to_all_its_sketch_elements.end());
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x002).first_element_, 0);
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x002).block_size_, 2);
        // AGC(2,2)
        ASSERT_NE(representation_to_all_its_sketch_elements.find(0x021), representation_to_all_its_sketch_elements.end());
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x021).first_element_, 2);
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x021).block_size_, 2);
        // ATC(4,1)
        ASSERT_NE(representation_to_all_its_sketch_elements.find(0x031), representation_to_all_its_sketch_elements.end());
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x031).first_element_, 4);
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x031).block_size_, 1);
        // ATG(5,1)
        ASSERT_NE(representation_to_all_its_sketch_elements.find(0x032), representation_to_all_its_sketch_elements.end());
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x032).first_element_, 5);
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x032).block_size_, 1);
        // CAA(6,1)
        ASSERT_NE(representation_to_all_its_sketch_elements.find(0x100), representation_to_all_its_sketch_elements.end());
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x100).first_element_, 6);
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x100).block_size_, 1);
        // CTA(7,1)
        ASSERT_NE(representation_to_all_its_sketch_elements.find(0x130), representation_to_all_its_sketch_elements.end());
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x130).first_element_, 7);
        EXPECT_EQ(representation_to_all_its_sketch_elements.at(0x130).block_size_, 1);
    }

}
