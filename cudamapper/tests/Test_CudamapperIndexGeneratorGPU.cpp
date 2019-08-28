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
#include <utility>
#include "cudamapper_file_location.hpp"
#include "../src/index_generator_gpu.hpp"

namespace claragenomics
{

    void test_function(const std::string& filename,
                       const std::uint64_t minimizer_size,
                       const std::uint64_t window_size,
                       const std::uint64_t number_of_reads,
                       const std::vector<std::string>& read_id_to_read_name,
                       const std::vector<std::uint32_t> read_id_to_read_length,
                       const std::vector<std::pair<representation_t, std::vector<Minimizer>>>& representation_and_minimizers) {
        IndexGeneratorGPU index_generator(filename, minimizer_size, window_size);

        EXPECT_EQ(index_generator.minimizer_size(), minimizer_size);
        EXPECT_EQ(index_generator.window_size(), window_size);
        ASSERT_EQ(index_generator.number_of_reads(), number_of_reads);

        ASSERT_EQ(index_generator.read_id_to_read_name().size(), number_of_reads);
        for (std::size_t read_id; read_id < number_of_reads; ++read_id) {
            EXPECT_EQ(index_generator.read_id_to_read_name()[read_id], read_id_to_read_name[read_id]) << "read_id: " << read_id;
            EXPECT_EQ(index_generator.read_id_to_read_length()[read_id], read_id_to_read_length[read_id]) << "read_id: " << read_id;
        }

        const std::vector<IndexGenerator::RepresentationAndSketchElements>& representations_and_sketch_elements = index_generator.representations_and_sketch_elements();
        ASSERT_EQ(representations_and_sketch_elements.size(), representation_and_minimizers.size());
        for(std::size_t i = 0; i < representation_and_minimizers.size(); ++i) {
            const representation_t representation_expected = representation_and_minimizers[i].first;
            const representation_t representation_generated = representations_and_sketch_elements[i].representation_;
            ASSERT_EQ(representation_expected, representation_generated) << "representation: " << representation_expected;

            const std::vector<Minimizer>& current_minimizers_expected = representation_and_minimizers[i].second;
            const std::vector<std::unique_ptr<SketchElement>>& current_minimizers_generated = representations_and_sketch_elements[i].sketch_elements_;
            ASSERT_EQ(current_minimizers_expected.size(), current_minimizers_generated.size()) << "representation: " << representation_expected;

            for (std::size_t j = 0; j < current_minimizers_expected.size(); ++j) {
                EXPECT_EQ(current_minimizers_generated[j]->representation(), current_minimizers_expected[j].representation()) << "representation: " << representation_expected << ", index: " << j;
                EXPECT_EQ(current_minimizers_generated[j]->position_in_read(), current_minimizers_expected[j].position_in_read()) << "representation: " << representation_expected << ", index: " << j;
                EXPECT_EQ(current_minimizers_generated[j]->direction(), current_minimizers_expected[j].direction()) << "representation: " << representation_expected << ", index: " << j;
                EXPECT_EQ(current_minimizers_generated[j]->read_id(), current_minimizers_expected[j].read_id()) << "representation: " << representation_expected << ", index: " << j;
            }
        }
    }

    TEST(TestCudamapperIndexGeneratorGPU, GATT_4_1) {
        // >read_0
        // GATT
        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta");
        std::uint64_t minimizer_size = 4;
        std::uint64_t window_size = 1;
        std::uint64_t number_of_reads = 1;

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_length.push_back(4);

        std::vector<std::pair<representation_t, std::vector<Minimizer>>> representation_and_minimizers;
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b1101, {}});
        representation_and_minimizers.back().second.push_back({0b1101, 0, SketchElement::DirectionOfRepresentation::REVERSE, 0});

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_and_minimizers);
    }

    TEST(TestCudamapperIndexGeneratorGPU, GATT_2_3) {
        // >read_0
        // GATT
        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta");
        std::uint64_t minimizer_size = 2;
        std::uint64_t window_size = 3;
        std::uint64_t number_of_reads = 1;

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_length.push_back(4);

        std::vector<std::pair<representation_t, std::vector<Minimizer>>> representation_and_minimizers;
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b0000, {}});
        representation_and_minimizers.back().second.push_back({0b0000, 2, SketchElement::DirectionOfRepresentation::REVERSE, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b0011, {}});
        representation_and_minimizers.back().second.push_back({0b0011, 1, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b1000, {}});
        representation_and_minimizers.back().second.push_back({0b1000, 0, SketchElement::DirectionOfRepresentation::FORWARD, 0});

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_and_minimizers);
    }

    TEST(TestCudamapperIndexGeneratorGPU, CCCATACC_2_8) {
        // *** Read is shorter than one full window, the result should be empty ***

        // >read_0
        // CCCATACC

        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/cccatacc.fasta");
        std::uint64_t minimizer_size = 2;
        std::uint64_t window_size = 8;
        std::uint64_t number_of_reads = 0; // the only read will be ignored as it does not fit a single window

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;

        std::vector<std::pair<representation_t, std::vector<Minimizer>>> representation_and_minimizers;

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_and_minimizers);
    }

    TEST(TestCudamapperIndexGeneratorGPU, CATCAAG_AAGCTA_3_5) {
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

        // all minimizers: (032, 0, R 0), (031, 1, F, 0), (002, 4, F, 0)
        // all minimizers sorted: (002,4,F,0), (031, 1, F, 0), (032, 0, R, 0)

        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "//catcaag_aagcta.fasta");
        std::uint64_t minimizer_size = 3;
        std::uint64_t window_size = 5;
        std::uint64_t number_of_reads = 1; // one read is ignored

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_length.push_back(7);

        std::vector<std::pair<representation_t, std::vector<Minimizer>>> representation_and_minimizers;
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b000010, {}});
        representation_and_minimizers.back().second.push_back({0b000010, 4, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b001101, {}});
        representation_and_minimizers.back().second.push_back({0b001101, 1, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b001110, {}});
        representation_and_minimizers.back().second.push_back({0b001110, 0, SketchElement::DirectionOfRepresentation::REVERSE, 0});

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_and_minimizers);



    }

    TEST(TestCudamapperIndexGeneratorGPU, CCCATAC_3_5) {
        // >read_0
        // CCCATACC

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

        // all minimizers: (111,0,F), (110,1,F), (032,2,R), (030,3,F), (011,5,F)
        // all minimizers sorted: (011,5,F), (030,3,F), (032,2,R), (110,1,F), (111,0,F)

        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/cccatacc.fasta");
        std::uint64_t minimizer_size = 3;
        std::uint64_t window_size = 5;
        std::uint64_t number_of_reads = 1;

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_length.push_back(8);

        std::vector<std::pair<representation_t, std::vector<Minimizer>>> representation_and_minimizers;
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b000101, {}});
        representation_and_minimizers.back().second.push_back({0b000101, 5, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b001100, {}});
        representation_and_minimizers.back().second.push_back({0b001100, 3, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b001110, {}});
        representation_and_minimizers.back().second.push_back({0b001110, 2, SketchElement::DirectionOfRepresentation::REVERSE, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b010100, {}});
        representation_and_minimizers.back().second.push_back({0b010100, 1, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b010101, {}});
        representation_and_minimizers.back().second.push_back({0b010101, 0, SketchElement::DirectionOfRepresentation::FORWARD, 0});

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_and_minimizers);

    }

    TEST(TestCudamapperIndexGeneratorGPU, CATCAAG_AAGCTA_3_2) {
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

        // all minimizers: (032,0,R,0), (031,1,F,0), (100,3,F,0), (002,4,F,0), (002,0,F,1), (021,2,R,1), (130,3,F,1)
        // all minimizers sorted: (002,4,F,0), (002,0,F,1), (021,2,R,1), (031,1,F,0), (032,0,R,0), (100,3,F,0), (130,3,F,1)

        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta");
        std::uint64_t minimizer_size = 3;
        std::uint64_t window_size = 2;
        std::uint64_t number_of_reads = 2;

        std::vector<std::string> read_id_to_read_name;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_name.push_back("read_1");

        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_length.push_back(7);
        read_id_to_read_length.push_back(6);

        std::vector<std::pair<representation_t, std::vector<Minimizer>>> representation_and_minimizers;

        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b000010, {}});
        representation_and_minimizers.back().second.push_back({0b000010, 4, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.back().second.push_back({0b000010, 0, SketchElement::DirectionOfRepresentation::FORWARD, 1});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b001001, {}});
        representation_and_minimizers.back().second.push_back({0b001001, 2, SketchElement::DirectionOfRepresentation::REVERSE, 1});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b001101, {}});
        representation_and_minimizers.back().second.push_back({0b001101, 1, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b001110, {}});
        representation_and_minimizers.back().second.push_back({0b001110, 0, SketchElement::DirectionOfRepresentation::REVERSE, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b010000, {}});
        representation_and_minimizers.back().second.push_back({0b010000, 3, SketchElement::DirectionOfRepresentation::FORWARD, 0});
        representation_and_minimizers.push_back(std::pair<representation_t, std::vector<Minimizer>>{0b011100, {}});
        representation_and_minimizers.back().second.push_back({0b011100, 3, SketchElement::DirectionOfRepresentation::FORWARD, 1});


        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_and_minimizers);

    }

}
