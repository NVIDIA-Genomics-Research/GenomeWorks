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
#include "../src/index_generator_gpu.hpp"

namespace claragenomics {

    TEST(TestCudamapperIndexCPU, OneReadOneMinimizer) {
        // >read_0
        // GATT

        // GATT = 0b10001111
        // AATC = 0b00001101 <- minimizer

        IndexGeneratorGPU index_generator(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta", 4, 1);
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

        const std::vector<std::vector<Index::RepresentationToSketchElements>>& read_id_and_representation_to_sketch_elements = index.read_id_and_representation_to_sketch_elements();
        ASSERT_EQ(read_id_and_representation_to_sketch_elements.size(), 1);
        const std::vector<Index::RepresentationToSketchElements>& rep_to_se_for_read_0 = read_id_and_representation_to_sketch_elements[0];
        ASSERT_EQ(rep_to_se_for_read_0.size(), 1);
        ASSERT_EQ(rep_to_se_for_read_0[0].representation_, 0b00001101);
        EXPECT_EQ(rep_to_se_for_read_0[0].sketch_elements_for_representation_and_read_id_.first_element_, 0);
        EXPECT_EQ(rep_to_se_for_read_0[0].sketch_elements_for_representation_and_read_id_.block_size_, 1);
        EXPECT_EQ(rep_to_se_for_read_0[0].sketch_elements_for_representation_and_all_read_ids_.first_element_, 0);
        EXPECT_EQ(rep_to_se_for_read_0[0].sketch_elements_for_representation_and_all_read_ids_.block_size_, 1);
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
        // AGCT: <AGC>, GCT, GCT, <AGC>  // only the last AGC is taken by IndexGeneratorGPU
        // GCTA: GCT, <AGC>, CTA, TAG
        // Front end minimizers: none
        // Back end miniminers: <CTA>, TAG
        // All minimizers: AAG(0f), AGC(1f), CTA(3f)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0         1         2         3         4         5         6
        // data arrays: AAG(4f0), AAG(0f1), AGC(2r1), ATC(1f0), ATG(0r0), CAA(3f0), CTA(3f1)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AAG(0,1)(0,2), ATC(3,1)(3,1), ATG(4,1)(4,1). CAA(5,1)(5,1))
        //                                                read_1(AAG(1,1)(0,2), AGC(2,1)(2,1), CTA(6,1)(6,1))

        IndexGeneratorGPU index_generator(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta", 3, 2);
        IndexCPU index(index_generator);

        ASSERT_EQ(index.number_of_reads(), 2);

        const std::vector<std::string>& read_id_to_read_name = index.read_id_to_read_name();
        ASSERT_EQ(read_id_to_read_name.size(), 2);
        EXPECT_EQ(read_id_to_read_name[0], std::string("read_0"));
        EXPECT_EQ(read_id_to_read_name[1], std::string("read_1"));

        const std::vector<position_in_read_t>& positions_in_reads = index.positions_in_reads();
        ASSERT_EQ(positions_in_reads.size(), 7);
        const std::vector<read_id_t>& read_ids = index.read_ids();
        ASSERT_EQ(read_ids.size(), 7);
        const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads = index.directions_of_reads();
        ASSERT_EQ(directions_of_reads.size(), 7);

        // Test data arrays

        // AAG(4f0)
        EXPECT_EQ(positions_in_reads[0], 4);
        EXPECT_EQ(read_ids[0], 0);
        EXPECT_EQ(directions_of_reads[0], SketchElement::DirectionOfRepresentation::FORWARD);
        // AAG(0f1)
        EXPECT_EQ(positions_in_reads[1], 0);
        EXPECT_EQ(read_ids[1], 1);
        EXPECT_EQ(directions_of_reads[1], SketchElement::DirectionOfRepresentation::FORWARD);
        // AGC(2r1)
        EXPECT_EQ(positions_in_reads[2], 2);
        EXPECT_EQ(read_ids[2], 1);
        EXPECT_EQ(directions_of_reads[2], SketchElement::DirectionOfRepresentation::REVERSE);
        // ATC(1f0)
        EXPECT_EQ(positions_in_reads[3], 1);
        EXPECT_EQ(read_ids[3], 0);
        EXPECT_EQ(directions_of_reads[3], SketchElement::DirectionOfRepresentation::FORWARD);
        // ATG(0r0)
        EXPECT_EQ(positions_in_reads[4], 0);
        EXPECT_EQ(read_ids[4], 0);
        EXPECT_EQ(directions_of_reads[4], SketchElement::DirectionOfRepresentation::REVERSE);
        // CAA(3f0)
        EXPECT_EQ(positions_in_reads[5], 3);
        EXPECT_EQ(read_ids[5], 0);
        EXPECT_EQ(directions_of_reads[5], SketchElement::DirectionOfRepresentation::FORWARD);
        // CTA(3f1)
        EXPECT_EQ(positions_in_reads[6], 3);
        EXPECT_EQ(read_ids[6], 1);
        EXPECT_EQ(directions_of_reads[6], SketchElement::DirectionOfRepresentation::FORWARD);

        // Test pointers

        const std::vector<std::vector<Index::RepresentationToSketchElements>>& read_id_and_representation_to_sketch_elements = index.read_id_and_representation_to_sketch_elements();
        ASSERT_EQ(read_id_and_representation_to_sketch_elements.size(), 2);

        // read_0(AAG(0,1)(0,2), ATC(3,1)(3,1), ATG(4,1)(4,1). CAA(5,1)(5,1))
        const std::vector<Index::RepresentationToSketchElements>& rep_to_se_for_read_0 = read_id_and_representation_to_sketch_elements[0];
        ASSERT_EQ(rep_to_se_for_read_0.size(), 4);
        ASSERT_EQ(rep_to_se_for_read_0[0].representation_, 0b000010); // AAG
        EXPECT_EQ(rep_to_se_for_read_0[0].sketch_elements_for_representation_and_read_id_.first_element_, 0);
        EXPECT_EQ(rep_to_se_for_read_0[0].sketch_elements_for_representation_and_read_id_.block_size_, 1);
        EXPECT_EQ(rep_to_se_for_read_0[0].sketch_elements_for_representation_and_all_read_ids_.first_element_, 0);
        EXPECT_EQ(rep_to_se_for_read_0[0].sketch_elements_for_representation_and_all_read_ids_.block_size_, 2);
        ASSERT_EQ(rep_to_se_for_read_0[1].representation_, 0b001101); // ATC
        EXPECT_EQ(rep_to_se_for_read_0[1].sketch_elements_for_representation_and_read_id_.first_element_, 3);
        EXPECT_EQ(rep_to_se_for_read_0[1].sketch_elements_for_representation_and_read_id_.block_size_, 1);
        EXPECT_EQ(rep_to_se_for_read_0[1].sketch_elements_for_representation_and_all_read_ids_.first_element_, 3);
        EXPECT_EQ(rep_to_se_for_read_0[1].sketch_elements_for_representation_and_all_read_ids_.block_size_, 1);
        ASSERT_EQ(rep_to_se_for_read_0[2].representation_, 0b001110); // ATG
        EXPECT_EQ(rep_to_se_for_read_0[2].sketch_elements_for_representation_and_read_id_.first_element_, 4);
        EXPECT_EQ(rep_to_se_for_read_0[2].sketch_elements_for_representation_and_read_id_.block_size_, 1);
        EXPECT_EQ(rep_to_se_for_read_0[2].sketch_elements_for_representation_and_all_read_ids_.first_element_, 4);
        EXPECT_EQ(rep_to_se_for_read_0[2].sketch_elements_for_representation_and_all_read_ids_.block_size_, 1);
        ASSERT_EQ(rep_to_se_for_read_0[3].representation_, 0b010000); // CAA
        EXPECT_EQ(rep_to_se_for_read_0[3].sketch_elements_for_representation_and_read_id_.first_element_, 5);
        EXPECT_EQ(rep_to_se_for_read_0[3].sketch_elements_for_representation_and_read_id_.block_size_, 1);
        EXPECT_EQ(rep_to_se_for_read_0[3].sketch_elements_for_representation_and_all_read_ids_.first_element_, 5);
        EXPECT_EQ(rep_to_se_for_read_0[3].sketch_elements_for_representation_and_all_read_ids_.block_size_, 1);

        // read_1(AAG(1,1)(0,2), AGC(2,1)(2,1), CTA(6,1)(6,1))
        const std::vector<Index::RepresentationToSketchElements>& rep_to_se_for_read_1 = read_id_and_representation_to_sketch_elements[1];
        ASSERT_EQ(rep_to_se_for_read_1.size(), 3);
        ASSERT_EQ(rep_to_se_for_read_1[0].representation_, 0b000010); // AAG
        EXPECT_EQ(rep_to_se_for_read_1[0].sketch_elements_for_representation_and_read_id_.first_element_, 1);
        EXPECT_EQ(rep_to_se_for_read_1[0].sketch_elements_for_representation_and_read_id_.block_size_, 1);
        EXPECT_EQ(rep_to_se_for_read_1[0].sketch_elements_for_representation_and_all_read_ids_.first_element_, 0);
        EXPECT_EQ(rep_to_se_for_read_1[0].sketch_elements_for_representation_and_all_read_ids_.block_size_, 2);
        ASSERT_EQ(rep_to_se_for_read_1[1].representation_, 0b001001); // AGC
        EXPECT_EQ(rep_to_se_for_read_1[1].sketch_elements_for_representation_and_read_id_.first_element_, 2);
        EXPECT_EQ(rep_to_se_for_read_1[1].sketch_elements_for_representation_and_read_id_.block_size_, 1);
        EXPECT_EQ(rep_to_se_for_read_1[1].sketch_elements_for_representation_and_all_read_ids_.first_element_, 2);
        EXPECT_EQ(rep_to_se_for_read_1[1].sketch_elements_for_representation_and_all_read_ids_.block_size_, 1);
        ASSERT_EQ(rep_to_se_for_read_1[2].representation_, 0b011100); // CTA
        EXPECT_EQ(rep_to_se_for_read_1[2].sketch_elements_for_representation_and_read_id_.first_element_, 6);
        EXPECT_EQ(rep_to_se_for_read_1[2].sketch_elements_for_representation_and_read_id_.block_size_, 1);
        EXPECT_EQ(rep_to_se_for_read_1[2].sketch_elements_for_representation_and_all_read_ids_.first_element_, 6);
        EXPECT_EQ(rep_to_se_for_read_1[2].sketch_elements_for_representation_and_all_read_ids_.block_size_, 1);
    }

}
