/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
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

        // All minimizers: ATG(0r0), ATC(1f0), AAG(4f0)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0         1         2
        // data arrays: AAG(4f0), ATC(1f0), ATG(0r0)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AAG(0,1)(0,1), ATC(1,1)(1,1), ATG(2,1)(2,1)

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
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001110, {2,1}, {2,1}}); // ATG

        std::vector<position_in_read_t> expected_positions_in_reads;
        std::vector<read_id_t> expected_read_ids;
        std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
        expected_positions_in_reads.push_back(4); // AAG(4f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(1); // ATC(1f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(0); // ATG(0r0)
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

        // All minimizers: CCC(0f), CCA(1f), ATG(2r), ATA(3f), ACC(5f)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0         1         2
        // data arrays: ACC(5f0), ATA(3f0), ATG(2r0), CCA(1f0), CCC(0f0)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AAC(0,1)(0,1), AAT(1,1)(1,1), ATG(2,1)(2,1), CCA(3,1)(3,1), CCC(4,1)(4,1)

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
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001110, {2,1}, {2,1}}); // ATG
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b010100, {3,1}, {3,1}}); // CCA
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
        expected_positions_in_reads.push_back(2); // ATG(2r0)
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

    TEST(TestCudamapperIndexGPU, AAAACTGAA_GCCAAAG_2_3) {
        // >read_0
        // AAAACTGAA
        // >read_1
        // GCCAAAG

        // ** AAAACTGAA **

        // kmer representation: forward, reverse
        // AA: <00> 33
        // AA: <00> 33
        // AA: <00> 33
        // AC: <01> 23
        // CT:  13 <02>
        // TG:  32 <10>
        // GA: <20> 31
        // AA: <00> 33

        // front end minimizers: representation, position_in_read, direction, read_id
        // AA : 00 0 F 0
        // AAA: 00 1 F 0

        // central minimizers
        // AAAA: 00 2 F 0
        // AAAC: 00 2 F 0
        // AACT: 00 2 F 0
        // ACTG: 01 3 F 0
        // CTGA: 02 4 R 0
        // TGAA: 00 7 F 0

        // back end minimizers
        // GAA: 00 7 F 0
        // AA : 00 7 F 0

        // All minimizers: AA(0f), AA(1f), AA(2f), AC(3f), AG(4r), AA (7f)

        // ** GCCAAAG **

        // kmer representation: forward, reverse
        // GC: <21> 21
        // CC: <11> 22
        // CA: <10> 32
        // AA: <00> 33
        // AA: <00> 33
        // AG: <03> 21

        // front end minimizers: representation, position_in_read, direction, read_id
        // GC : 21 0 F 0
        // GCC: 11 1 F 0

        // central minimizers
        // GCCA: 10 2 F 0
        // CCAA: 00 3 F 0
        // CAAA: 00 4 F 0
        // AAAG: 00 4 F 0

        // back end minimizers
        // AAG: 00 4 F 0
        // AG : 03 5 F 0

        // All minimizers: GC(0f), CC(1f), CA(2f), AA(3f), AA(4f), AG(5f)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0        1        2        3        4        5        6        7        8        9        10       11
        // data arrays: AA(0f0), AA(1f0), AA(2f0), AA(7f0), AA(3f1), AA(4f1), AC(3f0), AG(4r0), AG(5f1), CA(2f1), CC(1f1), GC(0f1)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AA(0,4)(0,6), AC(6,1)(6,1), AG(7,1)(7,2)
        //                                                read_1(AA(4,2)(0,6), AG(8,1)(7,2), CA(9,1)(9,1), CC(10,1)(10,1), GC(11,1)(11,1)

        const std::string filename = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/aaaactgaa_gccaaag.fasta";
        const std::uint64_t minimizer_size = 2;
        const std::uint64_t window_size = 3;

        const std::uint64_t expected_number_of_reads = 2;

        std::vector<std::string> expected_read_id_to_read_name;
        expected_read_id_to_read_name.push_back("read_0");
        expected_read_id_to_read_name.push_back("read_1");

        std::vector<std::uint32_t> expected_read_id_to_read_length;
        expected_read_id_to_read_length.push_back(9);
        expected_read_id_to_read_length.push_back(7);

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(2);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0000, { 0, 4}, { 0, 6}}); // AA
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0001, { 6, 1}, { 6, 1}}); // AC
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0010, { 7, 1}, { 7, 2}}); // AG
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b0000, { 4, 2}, { 0, 6}}); // AA
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b0010, { 8, 1}, { 7, 2}}); // AG
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b0100, { 9, 1}, { 9, 1}}); // CA
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b0101, {10, 1}, {10, 1}}); // CC
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b1001, {11, 1}, {11, 1}}); // GC

        std::vector<position_in_read_t> expected_positions_in_reads;
        std::vector<read_id_t> expected_read_ids;
        std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
        expected_positions_in_reads.push_back(0); // AA(0f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(1); // AA(1f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(2); // AA(2f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(7); // AA(7f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(3); // AA(3f1)
        expected_read_ids.push_back(1);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(4); // AA(4f1)
        expected_read_ids.push_back(1);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(3); // AC(3f0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(4); // AG(4r0)
        expected_read_ids.push_back(0);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
        expected_positions_in_reads.push_back(5); // AG(5f1)
        expected_read_ids.push_back(1);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(2); // CA(2f1)
        expected_read_ids.push_back(1);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(1); // CC(1f1)
        expected_read_ids.push_back(1);
        expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
        expected_positions_in_reads.push_back(0); // GC(0f1)
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

namespace details {

namespace index_gpu {

    // ************ Test representation_buckets **************

    TEST(TestCudamapperIndexGPU, representation_buckets_1) {
        // approximate_sketch_elements_per_bucket = 7
        // sample_length = 7 / 3 = 2
        //
        // (1 1 2 2 4 4 6 6 9 9)
        //  ^   ^   ^   ^   ^
        // (0 0 1 5 5 5 7 8 8 8)
        //  ^   ^   ^   ^   ^
        // (1 1 1 1 3 4 5 7 9 9)
        //  ^   ^   ^   ^   ^
        //
        // samples_in_one_bucket = 2 * 3 = 6
        // Sorted: 0 1 1 1 1 2 3 4 5 5 6 7 8 9 9
        //         ^     ^     ^     ^     ^

        std::vector<std::vector<representation_t>> arrays_of_representations;
        arrays_of_representations.push_back({{1, 1, 2, 2, 4, 4, 6, 6, 9, 9}});
        arrays_of_representations.push_back({{0, 0, 1, 5, 5, 5, 7, 8, 8, 8}});
        arrays_of_representations.push_back({{1, 1, 1, 1, 3, 4, 5, 7, 9, 9}});

        std::vector<representation_t> res = generate_representation_buckets(arrays_of_representations, 7);

        std::vector<representation_t> expected_res = {0, 1, 3, 5, 8};

        ASSERT_EQ(res.size(), expected_res.size());
        for (std::size_t i = 0; i < expected_res.size(); ++i) {
            EXPECT_EQ(res[i], expected_res[i]) << "index: " << i;
        }
    }

    TEST(TestCudamapperIndexGPU, representation_buckets_2) {
        // approximate_sketch_elements_per_bucket = 5
        // sample_length = 5 / 3 = 1
        //
        // (1 1 2 2 4 4 6 6 9 9)
        //  ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        // (0 0 1 5 5 5 7 8 8 8)
        //  ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        // (1 1 1 3 3 4 5 7 9 9)
        //  ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        //
        // samples_in_one_bucket = 5 / 1 = 5
        // Sorted: 0 0 1 1 1 1 1 1 2 2 3 3 4 4 4 5 5 5 5 6 6 7 7 8 8 8 9 9 9 9
        //         ^         ^         ^         ^         ^         ^

        std::vector<std::vector<representation_t>> arrays_of_representations;
        arrays_of_representations.push_back({{1, 1, 2, 2, 4, 4, 6, 6, 9, 9}});
        arrays_of_representations.push_back({{0, 0, 1, 5, 5, 5, 7, 8, 8, 8}});
        arrays_of_representations.push_back({{1, 1, 1, 3, 3, 4, 5, 7, 9, 9}});

        std::vector<representation_t> res = generate_representation_buckets(arrays_of_representations, 5);

        std::vector<representation_t> expected_res = {0, 1, 3, 5, 6, 8};

        ASSERT_EQ(res.size(), expected_res.size());
        for (std::size_t i = 0; i < expected_res.size(); ++i) {
            EXPECT_EQ(res[i], expected_res[i]) << "index: " << i;
        }
    }


    TEST(TestCudamapperIndexGPU, representation_buckets_3) {
        // approximate_sketch_elements_per_bucket = 3
        // sample_length = 3 / 3 = 1
        //
        // (1 1 2 2 4 4 6 6 9 9)
        //  ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        // (0 0 1 5 5 5 7 8 8 8)
        //  ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        // (1 1 1 3 3 4 5 7 9 9)
        //  ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        //
        // samples_in_one_bucket = 3 / 1 = 3
        // Sorted: 0 0 1 1 1 1 1 1 2 2 3 3 4 4 4 5 5 5 5 6 6 7 7 8 8 8 9 9 9 9
        //         ^     ^     ^     ^     ^     ^     ^     ^     ^     ^

        std::vector<std::vector<representation_t>> arrays_of_representations;
        arrays_of_representations.push_back({{1, 1, 2, 2, 4, 4, 6, 6, 9, 9}});
        arrays_of_representations.push_back({{0, 0, 1, 5, 5, 5, 7, 8, 8, 8}});
        arrays_of_representations.push_back({{1, 1, 1, 3, 3, 4, 5, 7, 9, 9}});

        std::vector<representation_t> res = generate_representation_buckets(arrays_of_representations, 3);

        std::vector<representation_t> expected_res = {0, 1, 2, 4, 5, 7, 8, 9};

        ASSERT_EQ(res.size(), expected_res.size());
        for (std::size_t i = 0; i < expected_res.size(); ++i) {
            EXPECT_EQ(res[i], expected_res[i]) << "index: " << i;
        }
    }

    TEST(TestCudamapperIndexGPU, representation_buckets_4) {
        // approximate_sketch_elements_per_bucket = 9
        // sample_length = 9 / 3 = 3
        //
        // (1 1 2 2 4 4 6 6 9 9)
        //  ^     ^     ^     ^
        // (0 0 1 5 5 5 7 8 8 8)
        //  ^     ^     ^     ^
        // (1 1 1 3 3 4 5 7 9 9)
        //  ^     ^     ^     ^
        //
        // samples_in_one_bucket = 9 / 3 = 3
        // Sorted: 0 1 1 2 3 5 5 6 7 8 9 9
        //         ^     ^     ^     ^

        std::vector<std::vector<representation_t>> arrays_of_representations;
        arrays_of_representations.push_back({{1, 1, 2, 2, 4, 4, 6, 6, 9, 9}});
        arrays_of_representations.push_back({{0, 0, 1, 5, 5, 5, 7, 8, 8, 8}});
        arrays_of_representations.push_back({{1, 1, 1, 3, 3, 4, 5, 7, 9, 9}});

        std::vector<representation_t> res = generate_representation_buckets(arrays_of_representations, 9);

        std::vector<representation_t> expected_res = {0, 2, 5, 8};

        ASSERT_EQ(res.size(), expected_res.size());
        for (std::size_t i = 0; i < expected_res.size(); ++i) {
            EXPECT_EQ(res[i], expected_res[i]) << "index: " << i;
        }
    }

    TEST(TestCudamapperIndexGPU, representation_buckets_5) {
        // approximate_sketch_elements_per_bucket = 9
        // sample_length = 9 / 3 = 3
        //
        // (1 1 2)
        //  ^
        // (0 0 1)
        //  ^
        // (1 1 1)
        //  ^
        //
        // samples_in_one_bucket = 9 / 3 = 3
        // Sorted: 0 1 1
        //         ^

        std::vector<std::vector<representation_t>> arrays_of_representations;
        arrays_of_representations.push_back({{1, 1, 2}});
        arrays_of_representations.push_back({{0, 0, 1}});
        arrays_of_representations.push_back({{1, 1, 1}});

        std::vector<representation_t> res = generate_representation_buckets(arrays_of_representations, 9);

        std::vector<representation_t> expected_res = {0};

        ASSERT_EQ(res.size(), expected_res.size());
        for (std::size_t i = 0; i < expected_res.size(); ++i) {
            EXPECT_EQ(res[i], expected_res[i]) << "index: " << i;
        }
    }

    TEST(TestCudamapperIndexGPU, representation_buckets_exception) {
        // approximate_sketch_elements_per_bucket is smaller than the number of arrays -> function throws

        std::vector<std::vector<representation_t>> arrays_of_representations;
        arrays_of_representations.push_back({{1, 1, 2}});
        arrays_of_representations.push_back({{0, 0, 1}});
        arrays_of_representations.push_back({{1, 1, 1}});

        EXPECT_NO_THROW(generate_representation_buckets(arrays_of_representations, 3));

        EXPECT_THROW(generate_representation_buckets(arrays_of_representations, 2),
                                                     approximate_sketch_elements_per_bucket_too_small
                                                    );
    }

    // ************ Test representation_iterators **************
    TEST(TestCudamapperIndexGPU, representation_iterators) {
        std::vector<std::vector<representation_t>> arrays_of_representations;
        //                                    0  1  2  3  4  5  6  7  8  9 10 11 12 13
        arrays_of_representations.push_back({{1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 7, 8, 9, 9}});
        arrays_of_representations.push_back({{0, 0, 0, 3, 3, 5, 5, 5, 6, 7, 7}});
        arrays_of_representations.push_back({{6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9}});

        auto res = generate_representation_indices(arrays_of_representations, 0);
        EXPECT_EQ(res[0], 0u);
        EXPECT_EQ(res[1], 0u);
        EXPECT_EQ(res[2], 0u);

        res = generate_representation_indices(arrays_of_representations, 1);
        EXPECT_EQ(res[0], 0u);
        EXPECT_EQ(res[1], 3u);
        EXPECT_EQ(res[2], 0u);

        res = generate_representation_indices(arrays_of_representations, 6);
        EXPECT_EQ(res[0], 9u);
        EXPECT_EQ(res[1], 8u);
        EXPECT_EQ(res[2], 0u);

        res = generate_representation_indices(arrays_of_representations, 7);
        EXPECT_EQ(res[0], 10u);
        EXPECT_EQ(res[1], 9u);
        EXPECT_EQ(res[2], 1u);

        res = generate_representation_indices(arrays_of_representations, 8);
        EXPECT_EQ(res[0], 11u);
        EXPECT_EQ(res[1], 11u);
        EXPECT_EQ(res[2], 5u);

        res = generate_representation_indices(arrays_of_representations, 9);
        EXPECT_EQ(res[0], 12u);
        EXPECT_EQ(res[1], 11u);
        EXPECT_EQ(res[2], 8u);

        res = generate_representation_indices(arrays_of_representations, 10);
        EXPECT_EQ(res[0], 14u);
        EXPECT_EQ(res[1], 11u);
        EXPECT_EQ(res[2], 12u);
    }

    // ************ Test generate_bucket_boundary_indices **************

    void test_generate_bucket_boundary_indices(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                               const std::vector<representation_t>& representation_buckets,
                                               const std::vector<std::vector<std::pair<std::size_t, std::size_t>>>& expected_bucket_boundary_indices
                                              )
    {
        const std::size_t number_of_arrays = arrays_of_representations.size();
        const std::size_t number_of_buckets = representation_buckets.size();

        const std::vector<std::vector<std::pair<std::size_t, std::size_t>>> bucket_boundary_indices = generate_bucket_boundary_indices(arrays_of_representations, representation_buckets);
        ASSERT_EQ(bucket_boundary_indices.size(), number_of_buckets);

        for (std::size_t bucket_index = 0; bucket_index < number_of_buckets; ++bucket_index) {
            ASSERT_EQ(bucket_boundary_indices[bucket_index].size(), number_of_arrays) << "bucket: " << bucket_index;

            for (std::size_t array_index = 0; array_index < number_of_arrays; ++array_index) {
                EXPECT_EQ(bucket_boundary_indices[bucket_index][array_index].first,
                          expected_bucket_boundary_indices[bucket_index][array_index].first) << "bucket: " << bucket_index << ", array: " << array_index;
                EXPECT_EQ(bucket_boundary_indices[bucket_index][array_index].second,
                          expected_bucket_boundary_indices[bucket_index][array_index].second) << "bucket: " << bucket_index << ", array: " << array_index;
            }

        }
    }

    TEST(TestCudamapperIndexGPU, generate_bucket_boundary_indices) {
        std::vector<std::vector<representation_t>> arrays_of_representations;
        //                                    0  1  2  3  4  5  6  7  8  9 10 11 12 13
        arrays_of_representations.push_back({{1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 7, 8, 9, 9}});
        arrays_of_representations.push_back({{0, 0, 0, 3, 3, 5, 5, 5, 6, 7, 7}});
        arrays_of_representations.push_back({{6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9}});

        std::vector<representation_t> representation_buckets = {0, 5, 7, 9};

        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> expected_bucket_boundary_indices;
        expected_bucket_boundary_indices.push_back({{ 0, 9}, { 0, 5}, {0, 0}});
        expected_bucket_boundary_indices.push_back({{ 9,10}, { 5, 9}, {0, 1}});
        expected_bucket_boundary_indices.push_back({{10,12}, { 9,11}, {1, 8}});
        expected_bucket_boundary_indices.push_back({{12,14}, {11,11}, {8,12}});

        test_generate_bucket_boundary_indices(arrays_of_representations,
                                              representation_buckets,
                                              expected_bucket_boundary_indices
                                             );
    }

    // ************ Test merge_sketch_element_arrays **************

    template <typename ReadidPositionDirection>
    void test_merge_sketch_element_arrays(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                          const std::vector<std::vector<ReadidPositionDirection>>& arrays_of_readids_positions_directions,
                                          const std::uint64_t available_device_memory_bytes,
                                          const std::vector<representation_t>& expected_merged_representations,
                                          const std::vector<ReadidPositionDirection>& expected_merged_readids_positions_directions
                                        )
    {
        std::vector<representation_t> generated_merged_representations;
        std::vector<ReadidPositionDirection> generated_merged_readids_positions_directions;

        merge_sketch_element_arrays(arrays_of_representations,
                                    arrays_of_readids_positions_directions,
                                    available_device_memory_bytes,
                                    generated_merged_representations,
                                    generated_merged_readids_positions_directions
                                   );

        ASSERT_EQ(generated_merged_representations.size(), expected_merged_representations.size()) << "available_memory: " << available_device_memory_bytes;
        ASSERT_EQ(generated_merged_readids_positions_directions.size(), expected_merged_readids_positions_directions.size()) << "available_memory: " << available_device_memory_bytes;
        ASSERT_EQ(generated_merged_representations.size(), generated_merged_readids_positions_directions.size()) << "available_memory: " << available_device_memory_bytes;

        for (std::size_t i = 0; i < expected_merged_representations.size(); ++i) {
            ASSERT_EQ(generated_merged_representations[i], expected_merged_representations[i]) << "available_memory: " << available_device_memory_bytes << ", index: " << i;
            ASSERT_EQ(generated_merged_readids_positions_directions[i].read_id_, expected_merged_readids_positions_directions[i].read_id_) << "available_memory: " << available_device_memory_bytes << ", index: " << i;
            ASSERT_EQ(generated_merged_readids_positions_directions[i].position_in_read_, expected_merged_readids_positions_directions[i].position_in_read_) << "available_memory: " << available_device_memory_bytes << ", index: " << i;
            ASSERT_EQ(generated_merged_readids_positions_directions[i].direction_, expected_merged_readids_positions_directions[i].direction_) << "available_memory: " << available_device_memory_bytes << ", index: " << i;
        }
    }

    TEST(TestCudamapperIndexGPU, merge_sketch_element_arrays) {
        std::vector<std::vector<representation_t>> arrays_of_representations;
        //                                    0  1  2  3  4  5  6  7  8  9 10 11 12 13
        arrays_of_representations.push_back({{1, 1, 2, 2, 2, 3, 3, 3, 4, 6, 7, 8, 9, 9}});
        arrays_of_representations.push_back({{0, 0, 0, 3, 3, 5, 5, 5, 6, 7, 7}});
        arrays_of_representations.push_back({{6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9}});

        std::vector<std::vector<Minimizer::ReadidPositionDirection>> arrays_of_readids_positions_directions;
        arrays_of_readids_positions_directions.push_back({{0, 1, 1}, // 1
                                                          {0, 2, 0}, // 1
                                                          {4, 7, 1}, // 2
                                                          {5, 5, 1}, // 2
                                                          {6, 9, 1}, // 2
                                                          {1, 2, 0}, // 3
                                                          {2, 8, 1}, // 3
                                                          {4, 6, 1}, // 3
                                                          {5, 8, 1}, // 4
                                                          {3, 2, 0}, // 6
                                                          {8, 1, 0}, // 7
                                                          {0, 4, 1}, // 8
                                                          {2, 7, 0}, // 9
                                                          {2, 9, 0}, // 9
                                                         }
                                                        );
        arrays_of_readids_positions_directions.push_back({{10, 7, 0}, // 0
                                                          {10, 9, 0}, // 0
                                                          {12, 2, 1}, // 0
                                                          {13, 4, 1}, // 3
                                                          {15, 1, 1}, // 3
                                                          {12, 4, 1}, // 5
                                                          {13, 3, 0}, // 5
                                                          {13, 7, 0}, // 5
                                                          {14, 8, 1}, // 6
                                                          {15, 6, 1}, // 7
                                                          {15, 7, 0}, // 7
                                                         }
                                                        );
        arrays_of_readids_positions_directions.push_back({{25, 5, 0}, // 6
                                                          {26, 7, 0}, // 7
                                                          {26, 9, 1}, // 7
                                                          {27, 1, 1}, // 7
                                                          {27, 2, 1}, // 7
                                                          {20, 3, 1}, // 8
                                                          {20, 5, 0}, // 8
                                                          {20, 7, 1}, // 8
                                                          {20, 2, 0}, // 9
                                                          {20, 4, 0}, // 9
                                                          {20, 6, 1}, // 9
                                                          {20, 8, 1}, // 9
                                                         }
                                                        );

        std::vector<representation_t> expected_merged_representations = {0, 0, 0, 1, 1, 2, 2, 2, 3, 3,
                                                                         3, 3, 3, 4, 5, 5, 5, 6, 6, 6,
                                                                         7, 7, 7, 7, 7, 7, 7, 8, 8, 8,
                                                                         8, 9, 9, 9, 9, 9, 9};
        std::vector<Minimizer::ReadidPositionDirection> expected_merged_readids_positions_directions;
        expected_merged_readids_positions_directions.push_back({10, 7, 0}); // 0
        expected_merged_readids_positions_directions.push_back({10, 9, 0}); // 0
        expected_merged_readids_positions_directions.push_back({12, 2, 1}); // 0
        expected_merged_readids_positions_directions.push_back({ 0, 1, 1}); // 1
        expected_merged_readids_positions_directions.push_back({ 0, 2, 0}); // 1
        expected_merged_readids_positions_directions.push_back({ 4, 7, 1}); // 2
        expected_merged_readids_positions_directions.push_back({ 5, 5, 1}); // 2
        expected_merged_readids_positions_directions.push_back({ 6, 9, 1}); // 2
        expected_merged_readids_positions_directions.push_back({ 1, 2, 0}); // 3
        expected_merged_readids_positions_directions.push_back({ 2, 8, 1}); // 3
        expected_merged_readids_positions_directions.push_back({ 4, 6, 1}); // 3
        expected_merged_readids_positions_directions.push_back({13, 4, 1}); // 3
        expected_merged_readids_positions_directions.push_back({15, 1, 1}); // 3
        expected_merged_readids_positions_directions.push_back({ 5, 8, 1}); // 4
        expected_merged_readids_positions_directions.push_back({12, 4, 1}); // 5
        expected_merged_readids_positions_directions.push_back({13, 3, 0}); // 5
        expected_merged_readids_positions_directions.push_back({13, 7, 0}); // 5
        expected_merged_readids_positions_directions.push_back({ 3, 2, 0}); // 6
        expected_merged_readids_positions_directions.push_back({14, 8, 1}); // 6
        expected_merged_readids_positions_directions.push_back({25, 5, 0}); // 6
        expected_merged_readids_positions_directions.push_back({ 8, 1, 0}); // 7
        expected_merged_readids_positions_directions.push_back({15, 6, 1}); // 7
        expected_merged_readids_positions_directions.push_back({15, 7, 0}); // 7
        expected_merged_readids_positions_directions.push_back({26, 7, 0}); // 7
        expected_merged_readids_positions_directions.push_back({26, 9, 1}); // 7
        expected_merged_readids_positions_directions.push_back({27, 1, 1}); // 7
        expected_merged_readids_positions_directions.push_back({27, 2, 1}); // 7
        expected_merged_readids_positions_directions.push_back({ 0, 4, 1}); // 8
        expected_merged_readids_positions_directions.push_back({20, 3, 1}); // 8
        expected_merged_readids_positions_directions.push_back({20, 5, 0}); // 8
        expected_merged_readids_positions_directions.push_back({20, 7, 1}); // 8
        expected_merged_readids_positions_directions.push_back({ 2, 7, 0}); // 9
        expected_merged_readids_positions_directions.push_back({ 2, 9, 0}); // 9
        expected_merged_readids_positions_directions.push_back({20, 2, 0}); // 9
        expected_merged_readids_positions_directions.push_back({20, 4, 0}); // 9
        expected_merged_readids_positions_directions.push_back({20, 6, 1}); // 9
        expected_merged_readids_positions_directions.push_back({20, 8, 1}); // 9

        // all elements fit in one sort call
        test_merge_sketch_element_arrays(arrays_of_representations,
                                         arrays_of_readids_positions_directions,
                                         10000000,
                                         expected_merged_representations,
                                         expected_merged_readids_positions_directions
                                       );

        std::size_t element_size = sizeof(representation_t) + sizeof(Minimizer::ReadidPositionDirection);
        std::size_t data_in_bytes = expected_merged_representations.size() * element_size;

        // merge_sketch_element_arrays needs 2.1*data_in_bytes memory, so passing merge_sketch_element_arrays as available memory will cause it to chunk the merging process
        test_merge_sketch_element_arrays(arrays_of_representations,
                                         arrays_of_readids_positions_directions,
                                         data_in_bytes,
                                         expected_merged_representations,
                                         expected_merged_readids_positions_directions
                                       );

        // a really small amount of memory
        test_merge_sketch_element_arrays(arrays_of_representations,
                                         arrays_of_readids_positions_directions,
                                         200,
                                         expected_merged_representations,
                                         expected_merged_readids_positions_directions
                                       );

        // amount memory too small to do the merge
        EXPECT_THROW(test_merge_sketch_element_arrays(arrays_of_representations,
                                                      arrays_of_readids_positions_directions,
                                                      100,
                                                      expected_merged_representations,
                                                      expected_merged_readids_positions_directions
                                                    ),
                    approximate_sketch_elements_per_bucket_too_small
                   );
    }

    // ************ Test generate_sections_for_multithreaded_index_building **************
    void test_generate_sections_for_multithreaded_index_building(const std::vector<representation_t>& input_representations,
                                                                 const std::vector<std::pair<std::size_t, std::size_t>>& expected_sections
                                                                )
    {
        auto generated_sections = generate_sections_for_multithreaded_index_building(input_representations);

        ASSERT_EQ(generated_sections.size(), expected_sections.size()) << "std::thread::hardware_concurrency: " << std::thread::hardware_concurrency();

        for (std::size_t i = 0; i < generated_sections.size(); ++i) {
            EXPECT_EQ(generated_sections[i].first, expected_sections[i].first) << "std::thread::hardware_concurrency: " << std::thread::hardware_concurrency() << ", index: " << i;
            EXPECT_EQ(generated_sections[i].second, expected_sections[i].second) << "std::thread::hardware_concurrency: " << std::thread::hardware_concurrency() << ", index: " << i;
        }
    }

    TEST(TestCudamapperIndexGPU, generate_sections_for_multithreaded_index_building_1) {
        // 0 0 1 1 2 2 3 3 ...
        // ^   ^   ^   ^
        // Perfect case, every section has the same number of elements

        auto number_of_threads = std::thread::hardware_concurrency();
        number_of_threads = std::max(1u, number_of_threads);

        std::vector<representation_t> representations;
        std::vector<std::pair<std::size_t, std::size_t>> expected_sections;
        for (std::size_t thread_id = 0; thread_id < number_of_threads; ++thread_id) {
            representations.push_back(thread_id);
            representations.push_back(thread_id);
            expected_sections.push_back({2*thread_id, 2*(thread_id+1)});
        }

        test_generate_sections_for_multithreaded_index_building(representations,
                                                                expected_sections
                                                               );
    }

    TEST(TestCudamapperIndexGPU, generate_sections_for_multithreaded_index_building_2) {
        // 0 0 0 1 1 2 2 2 3 3 4 4 4 5 5
        // number_of_thread = 6
        // number_of_elements = 15
        // elements_per_section = 15/6 = 2
        //
        // * section 0 *
        // 0 0 0 1 1 2 2 2 3 3 4 4 4 5 5
        // ^   ^
        // after looking for upper bound for the element left of past_the_last
        // 0 0 0 1 1 2 2 2 3 3 4 4 4 5 5
        // ^     ^
        //
        // * section 1 *
        // 0 0 0 1 1 2 2 2 3 3 4 4 4 5 5
        //       ^   ^
        // after looking for upper bound for the element left of past_the_last
        // 0 0 0 1 1 2 2 2 3 3 4 4 4 5 5
        //       ^   ^
        //
        // * section 2 *
        // 0 0 0 1 1 2 2 2 3 3 4 4 4 5 5
        //           ^   ^
        // after looking for upper bound for the element left of past_the_last
        // 0 0 0 1 1 2 2 2 3 3 4 4 4 5 5
        //           ^     ^
        // ...

        auto number_of_threads = std::thread::hardware_concurrency();
        number_of_threads = std::max(1u, number_of_threads);

        std::vector<representation_t> representations;
        std::vector<std::pair<std::size_t, std::size_t>> expected_sections;
        for (std::size_t thread_id = 0; thread_id < number_of_threads; ++thread_id) {
            representations.push_back(thread_id);
            representations.push_back(thread_id);
            if (thread_id % 2 == 0) representations.push_back(thread_id);

            std::size_t first_element = 0;
            if (thread_id != 0) first_element = expected_sections.back().second;
            if (thread_id % 2 == 0) expected_sections.push_back({first_element, first_element + 3});
            else expected_sections.push_back({first_element, first_element + 2});
        }

        test_generate_sections_for_multithreaded_index_building(representations,
                                                                expected_sections
                                                               );
    }

    TEST(TestCudamapperIndexGPU, generate_sections_for_multithreaded_index_building_3) {
        // only one representation -> all threads except for the first one get no sections

        auto number_of_threads = std::thread::hardware_concurrency();
        number_of_threads = std::max(1u, number_of_threads);

        std::vector<representation_t> representations(2*number_of_threads, 0);
        std::vector<std::pair<std::size_t, std::size_t>> expected_sections;
        expected_sections.push_back({0, 2*number_of_threads});

        test_generate_sections_for_multithreaded_index_building(representations,
                                                                expected_sections
                                                               );
    }

    TEST(TestCudamapperIndexGPU, generate_sections_for_multithreaded_index_building_4) {
        // only two representation -> all threads except for the first one get no sections

        auto number_of_threads = std::thread::hardware_concurrency();
        number_of_threads = std::max(1u, number_of_threads);

        if (number_of_threads <= 2u) {return;
            std::cout << "Only " << number_of_threads << " threads, no need to execute this test";
        }

        std::vector<representation_t> representations(2*number_of_threads);
        std::fill(std::begin(representations), std::begin(representations) + number_of_threads, 0);
        std::fill(std::begin(representations) + number_of_threads, std::end(representations), 1);
        std::vector<std::pair<std::size_t, std::size_t>> expected_sections;
        expected_sections.push_back({0, number_of_threads});
        expected_sections.push_back({number_of_threads, 2*number_of_threads});

        test_generate_sections_for_multithreaded_index_building(representations,
                                                                expected_sections
                                                               );
    }

    TEST(TestCudamapperIndexGPU, generate_sections_for_multithreaded_index_building_5) {
        // less elements in representation than threads

        auto number_of_threads = std::thread::hardware_concurrency();
        number_of_threads = std::max(1u, number_of_threads);

        if (number_of_threads <= 2u) {return;
            std::cout << "Only " << number_of_threads << " threads, no need to execute this test";
        }


        std::vector<representation_t> representations;
        representations.push_back(0); // only two elements
        representations.push_back(0);
        std::vector<std::pair<std::size_t, std::size_t>> expected_sections;
        expected_sections.push_back({0, 2});

        test_generate_sections_for_multithreaded_index_building(representations,
                                                                expected_sections
                                                               );
    }

    // ************ Test build_index **************

    template <typename ReadidPositionDirection>
    void test_build_index(const std::vector<representation_t>& input_representations,
                          const std::vector<ReadidPositionDirection>& input_readids_positions_directions,
                          const std::vector<std::vector<Index::RepresentationToSketchElements>>& expected_read_id_and_representation_to_sketch_elements) {
        std::uint64_t number_of_reads = expected_read_id_and_representation_to_sketch_elements.size();

        ASSERT_EQ(input_representations.size(), input_readids_positions_directions.size());

        std::vector<position_in_read_t> generated_positions_in_reads;
        std::vector<read_id_t> generated_read_ids;
        std::vector<Minimizer::DirectionOfRepresentation> generated_directions_of_reads;
        std::vector<std::vector<Index::RepresentationToSketchElements>> generated_read_id_and_representation_to_sketch_elements;

        build_index(number_of_reads,
                    input_representations,
                    input_readids_positions_directions,
                    generated_positions_in_reads,
                    generated_read_ids,
                    generated_directions_of_reads,
                    generated_read_id_and_representation_to_sketch_elements
                   );

        ASSERT_EQ(input_readids_positions_directions.size(), generated_positions_in_reads.size());
        ASSERT_EQ(input_readids_positions_directions.size(), generated_read_ids.size());
        ASSERT_EQ(input_readids_positions_directions.size(), generated_directions_of_reads.size());

        for (std::size_t i = 0; i < input_readids_positions_directions.size(); ++i) {
            EXPECT_EQ(input_readids_positions_directions[i].position_in_read_, generated_positions_in_reads[i]) << "index: " << i;
            EXPECT_EQ(input_readids_positions_directions[i].read_id_, generated_read_ids[i]) << "index: " << i;
            EXPECT_EQ(Minimizer::DirectionOfRepresentation(input_readids_positions_directions[i].direction_), generated_directions_of_reads[i]) << "index: " << i;
        }

        ASSERT_EQ(expected_read_id_and_representation_to_sketch_elements.size(), generated_read_id_and_representation_to_sketch_elements.size());
        for (std::size_t read_id = 0; read_id < expected_read_id_and_representation_to_sketch_elements.size(); ++read_id) {
            ASSERT_EQ(expected_read_id_and_representation_to_sketch_elements[read_id].size(), generated_read_id_and_representation_to_sketch_elements[read_id].size()) << "read id: " << read_id;
            //for (const auto& foo : generated_read_id_and_representation_to_sketch_elements[read_id]) std::cout << foo.representation_ << std::endl;
            for (std::size_t representation_index = 0; representation_index < expected_read_id_and_representation_to_sketch_elements[read_id].size(); ++representation_index) {
                const auto& expected_data = expected_read_id_and_representation_to_sketch_elements[read_id][representation_index];
                const auto& generated_data = generated_read_id_and_representation_to_sketch_elements[read_id][representation_index];
                // check representation
                EXPECT_EQ(expected_data.representation_, generated_data.representation_) << "read id: " << read_id << ", representation index: " << representation_index;
                // check sketch_elements_for_representation_and_read_id_
                EXPECT_EQ(expected_data.sketch_elements_for_representation_and_read_id_.first_element_,
                          generated_data.sketch_elements_for_representation_and_read_id_.first_element_) << "read id: " << read_id << ", representation index: " << representation_index;
                EXPECT_EQ(expected_data.sketch_elements_for_representation_and_read_id_.block_size_,
                          generated_data.sketch_elements_for_representation_and_read_id_.block_size_) << "read id: " << read_id << ", representation index: " << representation_index;
                // check sketch_elements_for_representation_and_all_read_ids_
                EXPECT_EQ(expected_data.sketch_elements_for_representation_and_all_read_ids_.first_element_,
                          generated_data.sketch_elements_for_representation_and_all_read_ids_.first_element_) << "read id: " << read_id << ", representation index: " << representation_index;
                EXPECT_EQ(expected_data.sketch_elements_for_representation_and_all_read_ids_.block_size_,
                          generated_data.sketch_elements_for_representation_and_all_read_ids_.block_size_) << "read id: " << read_id << ", representation index: " << representation_index;
            }
        }
    }

    TEST(TestCudamapperIndexGPU, build_index_GATT_4_1) {
        // >read_0
        // GATT

        // GATT = 0b10001111
        // AATC = 0b00001101 <- minimizer

        std::vector<representation_t> input_representations({{0b00001101}});
        std::vector<Minimizer::ReadidPositionDirection> input_readids_positions_directions({{0, 0, 1}});

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(1);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b00001101, {0, 1}, {0, 1}});

        test_build_index(input_representations,
                         input_readids_positions_directions,
                         expected_read_id_and_representation_to_sketch_elements);
    }

    TEST(TestCudamapperIndexGPU, build_index_GATT_2_3) {
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

        std::vector<representation_t> input_representations;
        input_representations.push_back(0b1000);
        input_representations.push_back(0b0011);
        input_representations.push_back(0b0000);
        std::vector<Minimizer::ReadidPositionDirection> input_readids_positions_directions;
        input_readids_positions_directions.push_back({0, 0, 0});
        input_readids_positions_directions.push_back({0, 1, 0});
        input_readids_positions_directions.push_back({0, 2, 1});

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(1);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b1000, {0, 1}, {0, 1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0011, {1, 1}, {1, 1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0000, {2, 1}, {2, 1}});

        test_build_index(input_representations,
                         input_readids_positions_directions,
                         expected_read_id_and_representation_to_sketch_elements);
    }

    TEST(TestCudamapperIndexGPU, build_index_CCCATACC_3_5) {
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

        // All minimizers: CCC(0f), CCA(1f), ATG(2r), ATA(3f), ACC(5f)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0         1         2
        // data arrays: ACC(5f0), ATA(3f0), ATG(2r0), CCA(1f0), CCC(0f0)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AAC(0,1)(0,1), AAT(1,1)(1,1), ATG(2,1)(2,1), CCA(3,1)(3,1), CCC(4,1)(4,1)

        std::vector<representation_t> input_representations;
        input_representations.push_back(0b000101); // ACC
        input_representations.push_back(0b001100); // ATA
        input_representations.push_back(0b001110); // ATG
        input_representations.push_back(0b010000); // CAA
        input_representations.push_back(0b010101); // CCC
        std::vector<Minimizer::ReadidPositionDirection> input_readids_positions_directions;
        input_readids_positions_directions.push_back({0, 5, 0}); // ACC
        input_readids_positions_directions.push_back({0, 3, 0}); // ATA
        input_readids_positions_directions.push_back({0, 2, 1}); // ATG
        input_readids_positions_directions.push_back({0, 1, 0}); // CCA
        input_readids_positions_directions.push_back({0, 0, 0}); // CCC

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(1);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b000101, {0,1}, {0,1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001100, {1,1}, {1,1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001110, {2,1}, {2,1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b010000, {3,1}, {3,1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b010101, {4,1}, {4,1}});

        test_build_index(input_representations,
                         input_readids_positions_directions,
                         expected_read_id_and_representation_to_sketch_elements);
    }

    TEST(TestCudamapperIndexGPU, build_index_CATCAAG_AAGCTA_3_2) {
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

        std::vector<representation_t> input_representations;
        input_representations.push_back(0b000010); // AAG
        input_representations.push_back(0b000010); // AAG
        input_representations.push_back(0b001001); // AGC
        input_representations.push_back(0b001101); // ATC
        input_representations.push_back(0b001110); // ATG
        input_representations.push_back(0b010000); // CAA
        input_representations.push_back(0b011100); // CTA
        std::vector<Minimizer::ReadidPositionDirection> input_readids_positions_directions;
        input_readids_positions_directions.push_back({0, 4, 0}); // AAG
        input_readids_positions_directions.push_back({1, 0, 0}); // AAG
        input_readids_positions_directions.push_back({1, 2, 1}); // AGC
        input_readids_positions_directions.push_back({0, 1, 0}); // ATC
        input_readids_positions_directions.push_back({0, 0, 1}); // ATG
        input_readids_positions_directions.push_back({0, 3, 0}); // CAA
        input_readids_positions_directions.push_back({1, 3, 0}); // CTA

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(2);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b000010, {0, 1}, {0, 2}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001101, {3, 1}, {3, 1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b001110, {4, 1}, {4, 1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b010000, {5, 1}, {5, 1}});
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b000010, {1, 1}, {0, 2}});
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b001001, {2, 1}, {2, 1}});
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b011100, {6, 1}, {6, 1}});

        test_build_index(input_representations,
                         input_readids_positions_directions,
                         expected_read_id_and_representation_to_sketch_elements);
    }

    TEST(TestCudamapperIndexGPU, build_index_AAAACTGAA_GCCAAAG_2_3) {
        // >read_0
        // AAAACTGAA
        // >read_1
        // GCCAAAG

        // ** AAAACTGAA **

        // kmer representation: forward, reverse
        // AA: <00> 33
        // AA: <00> 33
        // AA: <00> 33
        // AC: <01> 23
        // CT:  13 <02>
        // TG:  32 <10>
        // GA: <20> 31
        // AA: <00> 33

        // front end minimizers: representation, position_in_read, direction, read_id
        // AA : 00 0 F 0
        // AAA: 00 1 F 0

        // central minimizers
        // AAAA: 00 2 F 0
        // AAAC: 00 2 F 0
        // AACT: 00 2 F 0
        // ACTG: 01 3 F 0
        // CTGA: 02 4 R 0
        // TGAA: 00 7 F 0

        // back end minimizers
        // GAA: 00 7 F 0
        // AA : 00 7 F 0

        // All minimizers: AA(0f), AA(1f), AA(2f), AC(3f), AG(4r), AA (7f)

        // ** GCCAAAG **

        // kmer representation: forward, reverse
        // GC: <21> 21
        // CC: <11> 22
        // CA: <10> 32
        // AA: <00> 33
        // AA: <00> 33
        // AG: <03> 21

        // front end minimizers: representation, position_in_read, direction, read_id
        // GC : 21 0 F 0
        // GCC: 11 1 F 0

        // central minimizers
        // GCCA: 10 2 F 0
        // CCAA: 00 3 F 0
        // CAAA: 00 4 F 0
        // AAAG: 00 4 F 0

        // back end minimizers
        // AAG: 00 4 F 0
        // AG : 03 5 F 0

        // All minimizers: GC(0f), CC(1f), CA(2f), AA(3f), AA(4f), AG(5f)

        // (2r1) means position 2, reverse direction, read 1
        // (1,2) means array block start at element 1 and has 2 elements

        //              0        1        2        3        4        5        6        7        8        9        10       11
        // data arrays: AA(0f0), AA(1f0), AA(2f0), AA(7f0), AA(3f1), AA(4f1), AC(3f0), AG(4r0), AG(5f1), CA(2f1), CC(1f1), GC(0f1)
        //
        // read_1(AAG(1,1)(0,2)) means read_1 has "1" minimizer with representation AAG starting at position "1",
        // whereas in all reads there are "2" minimizers with representation AAG and they start at position "0"
        // read_id_and_representation_to_sketch_elements: read_0(AA(0,4)(0,6), AC(6,1)(6,1), AG(7,1)(7,2)
        //                                                read_1(AA(4,2)(0,6), AG(8,1)(7,2), CA(9,1)(9,1), CC(10,1)(10,1), GC(11,1)(11,1)

        std::vector<representation_t> input_representations;
        input_representations.push_back(0b0000); // AA
        input_representations.push_back(0b0000); // AA
        input_representations.push_back(0b0000); // AA
        input_representations.push_back(0b0000); // AA
        input_representations.push_back(0b0000); // AA
        input_representations.push_back(0b0000); // AA
        input_representations.push_back(0b0001); // AC
        input_representations.push_back(0b0010); // AG
        input_representations.push_back(0b0010); // AG
        input_representations.push_back(0b0100); // CA
        input_representations.push_back(0b0101); // CC
        input_representations.push_back(0b1001); // GC
        std::vector<Minimizer::ReadidPositionDirection> input_readids_positions_directions;
        input_readids_positions_directions.push_back({0, 0, 0}); // AA
        input_readids_positions_directions.push_back({0, 1, 0}); // AA
        input_readids_positions_directions.push_back({0, 2, 0}); // AA
        input_readids_positions_directions.push_back({0, 7, 0}); // AA
        input_readids_positions_directions.push_back({1, 3, 0}); // AA
        input_readids_positions_directions.push_back({1, 4, 0}); // AA
        input_readids_positions_directions.push_back({0, 3, 0}); // AC
        input_readids_positions_directions.push_back({0, 4, 1}); // AG
        input_readids_positions_directions.push_back({1, 5, 0}); // AG
        input_readids_positions_directions.push_back({1, 2, 0}); // CA
        input_readids_positions_directions.push_back({1, 1, 0}); // CC
        input_readids_positions_directions.push_back({1, 0, 0}); // GC

        std::vector<std::vector<Index::RepresentationToSketchElements>> expected_read_id_and_representation_to_sketch_elements(2);
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0000, { 0, 4}, { 0, 6}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0001, { 6, 1}, { 6, 1}});
        expected_read_id_and_representation_to_sketch_elements[0].push_back({0b0010, { 7, 1}, { 7, 2}});
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b0000, { 4, 2}, { 0, 6}});
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b0010, { 8, 1}, { 7, 2}});
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b0100, { 9, 1}, { 9, 1}});
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b0101, {10, 1}, {10, 1}});
        expected_read_id_and_representation_to_sketch_elements[1].push_back({0b1001, {11, 1}, {11, 1}});

        test_build_index(input_representations,
                         input_readids_positions_directions,
                         expected_read_id_and_representation_to_sketch_elements);
    }

} // namespace index_gpu
} // namespace details
} // namespace claragenomics
