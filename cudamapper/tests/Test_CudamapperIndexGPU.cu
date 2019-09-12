/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <math.h>

#include "gtest/gtest.h"
#include "cudamapper_file_location.hpp"
#include "../src/index_gpu.cuh"
#include "../src/minimizer.hpp"

namespace claragenomics {

    void test_function(const std::string& filename,
                       const std::uint64_t minimizer_size,
                       const std::uint64_t window_size,
                       const std::uint64_t expected_number_of_reads,
                       const std::vector<std::string>& expected_read_id_to_read_name,
                       const std::vector<std::uint32_t>& expected_read_id_to_read_length,
                       const std::vector<std::vector<Index::RepresentationToSketchElements>>& expected_read_id_and_representation_to_sketch_elements,
                       const std::vector<position_in_read_t>& expected_positions_in_reads,
                       const std::vector<read_id_t>& expected_read_ids,
                       const std::vector<SketchElement::DirectionOfRepresentation>& expected_directions_of_reads
                      )
    {
        IndexGPU<Minimizer> index(filename, minimizer_size, window_size);
        ASSERT_EQ(index.number_of_reads(), expected_number_of_reads);

        const std::vector<std::string>& read_id_to_read_name = index.read_id_to_read_name();
        ASSERT_EQ(read_id_to_read_name.size(), expected_read_id_to_read_name.size());
        ASSERT_EQ(read_id_to_read_name.size(), expected_number_of_reads);
        const std::vector<std::uint32_t>& read_id_to_read_length = index.read_id_to_read_length();
        ASSERT_EQ(read_id_to_read_length.size(), expected_read_id_to_read_length.size());
        ASSERT_EQ(read_id_to_read_length.size(), expected_number_of_reads);

        // check pointers to sections of arrays
        const std::vector<std::vector<Index::RepresentationToSketchElements>>& read_id_and_representation_to_sketch_elements = index.read_id_and_representation_to_sketch_elements();
        ASSERT_EQ(read_id_and_representation_to_sketch_elements.size(), expected_read_id_and_representation_to_sketch_elements.size());
        ASSERT_EQ(read_id_and_representation_to_sketch_elements.size(), expected_number_of_reads);
        for (std::size_t read_id = 0; read_id < expected_number_of_reads; ++read_id) {
            EXPECT_EQ(read_id_to_read_name[read_id], expected_read_id_to_read_name[read_id]) << "read_id: " << read_id;
            EXPECT_EQ(read_id_to_read_length[read_id], expected_read_id_to_read_length[read_id]) << "read_id: " << read_id;

            const std::vector<Index::RepresentationToSketchElements>& reps_to_se = read_id_and_representation_to_sketch_elements[read_id];
            const std::vector<Index::RepresentationToSketchElements>& exp_reps_to_se = expected_read_id_and_representation_to_sketch_elements[read_id];
            ASSERT_EQ(reps_to_se.size(), exp_reps_to_se.size()) << "read_id: " << read_id;

            for (std::size_t rep_id = 0; rep_id < exp_reps_to_se.size(); ++rep_id) {
                const Index::RepresentationToSketchElements& exp_rep_to_se = exp_reps_to_se[rep_id];
                const Index::RepresentationToSketchElements& rep_to_se = reps_to_se[rep_id];
                ASSERT_EQ(rep_to_se.representation_, exp_rep_to_se.representation_) << "read id: " << read_id << " , rep_id: " << rep_id;
                EXPECT_EQ(rep_to_se.sketch_elements_for_representation_and_read_id_.first_element_, exp_rep_to_se.sketch_elements_for_representation_and_read_id_.first_element_) << "read id: " << read_id << " , rep_id: " << rep_id;
                EXPECT_EQ(rep_to_se.sketch_elements_for_representation_and_read_id_.block_size_, exp_rep_to_se.sketch_elements_for_representation_and_read_id_.block_size_) << "read id: " << read_id << " , rep_id: " << rep_id;
                EXPECT_EQ(rep_to_se.sketch_elements_for_representation_and_all_read_ids_.first_element_, exp_rep_to_se.sketch_elements_for_representation_and_all_read_ids_.first_element_) << "read id: " << read_id << " , rep_id: " << rep_id;
                EXPECT_EQ(rep_to_se.sketch_elements_for_representation_and_all_read_ids_.block_size_, exp_rep_to_se.sketch_elements_for_representation_and_all_read_ids_.block_size_) << "read id: " << read_id << " , rep_id: " << rep_id;
            }
        }

        // check arrays
        const std::vector<position_in_read_t>& positions_in_reads = index.positions_in_reads();
        const std::vector<read_id_t>& read_ids = index.read_ids();
        const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads = index.directions_of_reads();
        ASSERT_EQ(positions_in_reads.size(), expected_positions_in_reads.size());
        ASSERT_EQ(read_ids.size(), expected_read_ids.size());
        ASSERT_EQ(directions_of_reads.size(), expected_directions_of_reads.size());
        ASSERT_EQ(positions_in_reads.size(), read_ids.size());
        ASSERT_EQ(positions_in_reads.size(), directions_of_reads.size());
        for (std::size_t i = 0; i < expected_positions_in_reads.size(); ++i) {
            EXPECT_EQ(positions_in_reads[i], expected_positions_in_reads[i]) << "i: " << i;
            EXPECT_EQ(read_ids[i], expected_read_ids[i]) << "i: " << i;
            EXPECT_EQ(directions_of_reads[i], expected_directions_of_reads[i]) << "i: " << i;
        }

        ASSERT_EQ(index.minimum_representation(), std::uint64_t(0));
        ASSERT_EQ(index.maximum_representation(), pow(4,std::uint64_t(minimizer_size))-1);
    }


    TEST(TestCudamapperIndexGPU, GATT_4_1) {
        // >read_0
        // GATT

        // GATT = 0b10001111
        // AATC = 0b00001101 <- minimizer

        const std::string filename = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta";
        const std::uint64_t minimizer_size = 4;
        const std::uint64_t window_size = 1;

        const std::uint64_t expected_number_of_reads = 1;

        std::vector<std::string> expected_read_id_to_read_name;
        expected_read_id_to_read_name.push_back("read_0");

        std::vector<std::uint32_t> expected_read_id_to_read_length;
        expected_read_id_to_read_length.push_back(4);

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(1);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b00001101, {0,1}, {0,1}});

        std::vector<position_in_read_t> expected_positions_in_reads;
        std::vector<read_id_t> expected_read_ids;
        std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
        expected_positions_in_reads.push_back(0);
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);

        test_function(filename,
                      minimizer_size,
                      window_size,
                      expected_number_of_reads,
                      expected_read_id_to_read_name,
                      expected_read_id_to_read_length,
                      expected_read_id_and_representation_to_sketch_elements,
                      expected_positions_in_reads,
                      expected_read_ids,
                      expected_directions_of_reads
                     );
    }

    TEST(TestCudamapperIndexGPU, GATT_2_3) {
        // >read_0
        // GATT

        // kmer representation: forward, reverse
        // GA: <20> 31
        // AT: <03> 03
        // TT:  33 <00>

        // front end minimizers: representation, position_in_read, direction, read_id
        // GA : 20 0 F 0
        // GAT: 03 1 F 0

        // central minimizers
        // GATT: 00 2 R 0

        // back end minimizers
        // ATT: 00 2 R 0
        // TT : 00 2 R 0

        // All minimizers: GA(0f), AT(1f), AA(2r)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0        1        2
        // data arrays: GA(0f0), AT(1f0), AA(2r0)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AA(0,1)(0,1), AT(1,1)(1,1), GA(2,1)(2,1)

        const std::string filename = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta";
        const std::uint64_t minimizer_size = 2;
        const std::uint64_t window_size = 3;

        const std::uint64_t expected_number_of_reads = 1;

        std::vector<std::string> expected_read_id_to_read_name;
        expected_read_id_to_read_name.push_back("read_0");

        std::vector<std::uint32_t> expected_read_id_to_read_length;
        expected_read_id_to_read_length.push_back(4);

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(1);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0000, {0,1}, {0,1}}); // AA
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0011, {1,1}, {1,1}}); // AT
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b1000, {2,1}, {2,1}}); // GA

        std::vector<position_in_read_t> expected_positions_in_reads;
        std::vector<read_id_t> expected_read_ids;
        std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
        expected_positions_in_reads.push_back(2); // AA(2r0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
        expected_positions_in_reads.push_back(1); // AT(1f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(0); // GA(0f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);

        test_function(filename,
                      minimizer_size,
                      window_size,
                      expected_number_of_reads,
                      expected_read_id_to_read_name,
                      expected_read_id_to_read_length,
                      expected_read_id_and_representation_to_sketch_elements,
                      expected_positions_in_reads,
                      expected_read_ids,
                      expected_directions_of_reads
                     );
    }


    TEST(TestCudamapperIndexGPU, CCCATACC_2_8) {
        // *** Read is shorter than one full window, the result should be empty ***

        // >read_0
        // CCCATACC

        const std::string filename = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/cccatacc.fasta";
        const std::uint64_t minimizer_size = 2;
        const std::uint64_t window_size = 8;

        const std::uint64_t expected_number_of_reads = 0;

        // all data arrays should be empty

        std::vector<std::string> expected_read_id_to_read_name;

        std::vector<std::uint32_t> expected_read_id_to_read_length;

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(0);

        std::vector<position_in_read_t> expected_positions_in_reads;
        std::vector<read_id_t> expected_read_ids;
        std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;

        test_function(filename,
                      minimizer_size,
                      window_size,
                      expected_number_of_reads,
                      expected_read_id_to_read_name,
                      expected_read_id_to_read_length,
                      expected_read_id_and_representation_to_sketch_elements,
                      expected_positions_in_reads,
                      expected_read_ids,
                      expected_directions_of_reads
                     );
    }

    TEST(TestCudamapperIndexGPU, CATCAAG_AAGCTA_3_5) {
        // *** One Read is shorter than one full window, the other is not ***

        // >read_0
        // CATCAAG
        // >read_1
        // AAGCTA

        // ** CATCAAG **

        // kmer representation: forward, reverse
        // CAT:  103 <032>
        // ATC: <031> 203
        // TCA: <310> 320
        // CAA: <100> 332
        // AAG: <002> 133

        // front end minimizers: representation, position_in_read, direction, read_id
        // CAT   : 032 0 R 0
        // CATC  : 031 1 F 0
        // CATCA : 031 1 F 0
        // CATCAA: 031 1 F 0

        // central minimizers
        // CATCAAG: 002 4 F 0

        // back end minimizers
        // ATCAAG: 002 4 F 0
        // TCAAG : 002 4 F 0
        // CAAG  : 002 4 F 0
        // AAG   : 002 4 F 0

        // ** AAGCTA **
        // ** read does not fit one array **

        // All minimizers: CAT(0r0), ATC(1f0), AAG(4f0)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0         1         2
        // data arrays: AAG(4f0), ATC(1f0), CAT(0r0)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AAG(0,1)(0,1), ATC(1,1)(1,1), CAT(2,1)(2,1)

        const std::string filename = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta";
        const std::uint64_t minimizer_size = 3;
        const std::uint64_t window_size = 5;

        const std::uint64_t expected_number_of_reads = 1;

        std::vector<std::string> expected_read_id_to_read_name;
        expected_read_id_to_read_name.push_back("read_0");

        std::vector<std::uint32_t> expected_read_id_to_read_length;
        expected_read_id_to_read_length.push_back(7);

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(1);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b000010, {0,1}, {0,1}}); // AAG
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001101, {1,1}, {1,1}}); // ATC
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001110, {2,1}, {2,1}}); // CAT

        std::vector<position_in_read_t> expected_positions_in_reads;
        std::vector<read_id_t> expected_read_ids;
        std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
        expected_positions_in_reads.push_back(4); // AAG(4f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(1); // ATC(1f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(0); // CAT(0r0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);

        test_function(filename,
                      minimizer_size,
                      window_size,
                      expected_number_of_reads,
                      expected_read_id_to_read_name,
                      expected_read_id_to_read_length,
                      expected_read_id_and_representation_to_sketch_elements,
                      expected_positions_in_reads,
                      expected_read_ids,
                      expected_directions_of_reads
                     );
    }

    TEST(TestCudamapperIndexGPU, CCCATACC_3_5) {
        // >read_0
        // CCCATACC

        // ** CCCATAC **

        // kmer representation: forward, reverse
        // CCC: <111> 222
        // CCA: <110> 322
        // CAT:  103 <032>
        // ATA: <030> 303
        // TAC:  301 <230>
        // ACC: <011> 223

        // front end minimizers: representation, position_in_read, direction
        // CCC   : 111 0 F
        // CCCA  : 110 1 F
        // CCCAT : 032 2 R
        // CCCATA: 030 3 F

        // central minimizers
        // CCCATAC: 030 3 F
        // CCATACC: 011 5 F

        // back end minimizers
        // CATACC: 011 5 F
        // ATACC : 011 5 F
        // TACC  : 011 5 F
        // ACC   : 011 5 F

        // All minimizers: CCC(0f), CAA(1f), CAT(2r), ATA(3f), ACC(5f)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0         1         2
        // data arrays: ACC(5f0), ATA(3f0), CAT(2r0), CAA(1f0), CCC(0f0)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AAC(0,1)(0,1), AAT(1,1)(1,1), CAT(2,1)(2,1), CAA(3,1)(3,1), CCC(4,1)(4,1)

        const std::string filename = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/cccatacc.fasta";
        const std::uint64_t minimizer_size = 3;
        const std::uint64_t window_size = 5;

        const std::uint64_t expected_number_of_reads = 1;

        std::vector<std::string> expected_read_id_to_read_name;
        expected_read_id_to_read_name.push_back("read_0");

        std::vector<std::uint32_t> expected_read_id_to_read_length;
        expected_read_id_to_read_length.push_back(8);

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(1);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b000101, {0,1}, {0,1}}); // ACC
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001100, {1,1}, {1,1}}); // ATA
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001110, {2,1}, {2,1}}); // CAT
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b010100, {3,1}, {3,1}}); // CAA
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b010101, {4,1}, {4,1}}); // CCC

        std::vector<position_in_read_t> expected_positions_in_reads;
        std::vector<read_id_t> expected_read_ids;
        std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
        expected_positions_in_reads.push_back(5); // ACC(5f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(3); // ATA(3f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(2); // CAT(2r0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
        expected_positions_in_reads.push_back(1); // CAA(1f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(0); // CCC(0f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);

        test_function(filename,
                      minimizer_size,
                      window_size,
                      expected_number_of_reads,
                      expected_read_id_to_read_name,
                      expected_read_id_to_read_length,
                      expected_read_id_and_representation_to_sketch_elements,
                      expected_positions_in_reads,
                      expected_read_ids,
                      expected_directions_of_reads
                     );
    }

    TEST(TestCudamapperIndexGPU, CATCAAG_AAGCTA_3_2) {
        // >read_0
        // CATCAAG
        // >read_1
        // AAGCTA

        // ** CATCAAG **

        // kmer representation: forward, reverse
        // CAT:  103 <032>
        // ATC: <031> 203
        // TCA: <310> 320
        // CAA: <100> 332
        // AAG: <002> 133

        // front end minimizers: representation, position_in_read, direction, read_id
        // CAT: 032 0 R 0

        // central minimizers
        // CATC: 031 1 F 0
        // ATCA: 031 1 F 0
        // TCAA: 100 3 F 0
        // CAAG: 002 4 F 0

        // back end minimizers
        // AAG: 002 4 F 0

        // All minimizers: ATC(1f), CAA(3f), AAG(4f), ATG(0r)

        // ** AAGCTA **

        // kmer representation: forward, reverse
        // AAG: <002> 133
        // AGC: <021> 213
        // GCT:  213 <021>
        // CTA: <130> 302

        // front end minimizers: representation, position_in_read, direction, read_id
        // AAG: 002 0 F 1

        // central minimizers
        // AAGC: 002 0 F 1
        // AGCT: 021 2 R 1 // only the last minimizer is saved
        // GCTA: 021 2 R 1

        // back end minimizers
        // CTA: 130 3 F 1
        // All minimizers: AAG(0f), AGC(1f), CTA(3f)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0         1         2         3         4         5         6
        // data arrays: AAG(4f0), AAG(0f1), AGC(2r1), ATC(1f0), ATG(0r0), CAA(3f0), CTA(3f1)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AAG(0,1)(0,2), ATC(3,1)(3,1), ATG(4,1)(4,1), CAA(5,1)(5,1))
        //                                                read_1(AAG(1,1)(0,2), AGC(2,1)(2,1), CTA(6,1)(6,1))


        const std::string filename = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta";
        const std::uint64_t minimizer_size = 3;
        const std::uint64_t window_size = 2;

        const std::uint64_t expected_number_of_reads = 2;

        std::vector<std::string> expected_read_id_to_read_name;
        expected_read_id_to_read_name.push_back("read_0");
        expected_read_id_to_read_name.push_back("read_1");

        std::vector<std::uint32_t> expected_read_id_to_read_length;
        expected_read_id_to_read_length.push_back(7);
        expected_read_id_to_read_length.push_back(6);

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(2);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b000010, {0,1}, {0,2}}); // AAG
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001101, {3,1}, {3,1}}); // ATC
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001110, {4,1}, {4,1}}); // ATG
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b010000, {5,1}, {5,1}}); // CAA
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b000010, {1,1}, {0,2}}); // AAG
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b001001, {2,1}, {2,1}}); // AGC
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b011100, {6,1}, {6,1}}); // CTA

        std::vector<position_in_read_t> expected_positions_in_reads;
        std::vector<read_id_t> expected_read_ids;
        std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
        expected_positions_in_reads.push_back(4); // AAG(4f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(0); // AAG(0f1)
        expected_read_ids.push_back(1);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(2); // AGC(2r1)
        expected_read_ids.push_back(1);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
        expected_positions_in_reads.push_back(1); // ATC(1f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(0); // ATG(0r0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
        expected_positions_in_reads.push_back(3); // CAA(3f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(3); // CTA(3f1)
        expected_read_ids.push_back(1);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);

        test_function(filename,
                      minimizer_size,
                      window_size,
                      expected_number_of_reads,
                      expected_read_id_to_read_name,
                      expected_read_id_to_read_length,
                      expected_read_id_and_representation_to_sketch_elements,
                      expected_positions_in_reads,
                      expected_read_ids,
                      expected_directions_of_reads
                     );
    }
}
