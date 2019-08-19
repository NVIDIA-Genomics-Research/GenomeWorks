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
#include "../src/index_generator_gpu.hpp"

namespace claragenomics
{

    void test_function(const std::string& filename,
                       const std::uint64_t minimizer_size,
                       const std::uint64_t window_size,
                       const std::uint64_t number_of_reads,
                       const std::vector<std::string>& read_id_to_read_name,
                       const std::vector<std::uint32_t> read_id_to_read_length,
                       const std::map<representation_t, std::vector<Minimizer>>& representation_to_minimizers) {
        IndexGeneratorGPU index_generator(filename, minimizer_size, window_size);

        EXPECT_EQ(index_generator.minimizer_size(), minimizer_size);
        EXPECT_EQ(index_generator.window_size(), window_size);
        ASSERT_EQ(index_generator.number_of_reads(), number_of_reads);

        ASSERT_EQ(index_generator.read_id_to_read_name().size(), number_of_reads);
        for (std::size_t read_id; read_id < number_of_reads; ++read_id) {
            EXPECT_EQ(index_generator.read_id_to_read_name()[read_id], read_id_to_read_name[read_id]) << "read_id: " << read_id;
            EXPECT_EQ(index_generator.read_id_to_read_length()[read_id], read_id_to_read_length[read_id]) << "read_id: " << read_id;
        }

        const auto& representation_to_sketch_elements = index_generator.representations_to_sketch_elements();
        ASSERT_EQ(representation_to_sketch_elements.size(), representation_to_minimizers.size());
        for(const auto& one_representation : representation_to_minimizers) {
            const representation_t current_representation = one_representation.first;
            const std::vector<Minimizer>& current_minimizers_expected = one_representation.second;
            ASSERT_NE(representation_to_sketch_elements.find(current_representation), std::end(representation_to_sketch_elements)) << "representation: " << current_representation;

            const std::vector<std::unique_ptr<SketchElement>>& current_minimizers_generated = (*representation_to_sketch_elements.find(current_representation)).second;
            ASSERT_EQ(current_minimizers_generated.size(), current_minimizers_expected.size()) << "representation: " << current_representation;

            for (std::size_t i = 0; i < current_minimizers_expected.size(); ++i) {
                EXPECT_EQ(current_minimizers_generated[i]->representation(), current_minimizers_expected[i].representation()) << "representation: " << current_representation << ", index: " << i;
                EXPECT_EQ(current_minimizers_generated[i]->position_in_read(), current_minimizers_expected[i].position_in_read()) << "representation: " << current_representation << ", index: " << i;
                EXPECT_EQ(current_minimizers_generated[i]->direction(), current_minimizers_expected[i].direction()) << "representation: " << current_representation << ", index: " << i;
                EXPECT_EQ(current_minimizers_generated[i]->read_id(), current_minimizers_expected[i].read_id()) << "representation: " << current_representation << ", index: " << i;
            }
        }
    }

    TEST(TestCudamapperIndexGeneratorGPU, GATT_4_1) {
        // >read_0
        // GATT
        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/one_read_one_minimizer.fasta");
        std::uint64_t minimizer_size = 4;
        std::uint64_t window_size = 1;
        std::uint64_t number_of_reads = 1;

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_length.push_back(4);

        std::map<representation_t, std::vector<Minimizer>> representation_to_minimizers;
        { // 0b1101
            std::vector<Minimizer> minimizers{{0b1101, 0, SketchElement::DirectionOfRepresentation::REVERSE, 0}};
            representation_to_minimizers[0b1101] = minimizers;
        }

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_to_minimizers);
    }

    TEST(TestCudamapperIndexGeneratorGPU, GATT_2_3) {
        // >read_0
        // GATT
        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/one_read_one_minimizer.fasta");
        std::uint64_t minimizer_size = 2;
        std::uint64_t window_size = 3;
        std::uint64_t number_of_reads = 1;

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_length.push_back(4);

        std::map<representation_t, std::vector<Minimizer>> representation_to_minimizers;
        { // 0b1000
            std::vector<Minimizer> minimizers{{0b1000, 0, SketchElement::DirectionOfRepresentation::FORWARD, 0}};
            representation_to_minimizers[0b1000] = minimizers;
        }
        { // 0b0011
            std::vector<Minimizer> minimizers{{0b0011, 1, SketchElement::DirectionOfRepresentation::FORWARD, 0}};
            representation_to_minimizers[0b0011] = minimizers;
        }
        { // 0b0000
            std::vector<Minimizer> minimizers{{0b0000, 2, SketchElement::DirectionOfRepresentation::REVERSE, 0}};
            representation_to_minimizers[0b0000] = minimizers;
        }

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_to_minimizers);
    }

    TEST(TestCudamapperIndexGeneratorGPU, CCCATACC_2_8) {
        // >read_0
        // CCCATACC

        // Read is shorter than one full window, the result should be empty
        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/cccatacc.fasta");
        std::uint64_t minimizer_size = 2;
        std::uint64_t window_size = 8;
        std::uint64_t number_of_reads = 0; // the only read will be ignored as it does not fit a single window

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;

        std::map<representation_t, std::vector<Minimizer>> representation_to_minimizers;

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_to_minimizers);
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

        // all minimizers: (111,0,F)m (110,1,F), (032,2,R), (030,3,F), (011,5,F)

        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/cccatacc.fasta");
        std::uint64_t minimizer_size = 3;
        std::uint64_t window_size = 5;
        std::uint64_t number_of_reads = 1;

        std::vector<std::string> read_id_to_read_name;
        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_length.push_back(8);

        std::map<representation_t, std::vector<Minimizer>> representation_to_minimizers;
        { // 111
            std::vector<Minimizer> minimizers{{0b010101, 0, SketchElement::DirectionOfRepresentation::FORWARD, 0}};
            representation_to_minimizers[0b010101] = minimizers;
        }
        { // 110
            std::vector<Minimizer> minimizers{{0b010100, 1, SketchElement::DirectionOfRepresentation::FORWARD, 0}};
            representation_to_minimizers[0b010100] = minimizers;
        }
        { // 032
            std::vector<Minimizer> minimizers{{0b001110, 2, SketchElement::DirectionOfRepresentation::REVERSE, 0}};
            representation_to_minimizers[0b001110] = minimizers;
        }
        { // 030
            std::vector<Minimizer> minimizers{{0b001100, 3, SketchElement::DirectionOfRepresentation::FORWARD, 0}};
            representation_to_minimizers[0b001100] = minimizers;
        }
        { // 011
            std::vector<Minimizer> minimizers{{0b000101, 5, SketchElement::DirectionOfRepresentation::FORWARD, 0}};
            representation_to_minimizers[0b000101] = minimizers;
        }

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_to_minimizers);

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

        std::string filename(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/two_reads_multiple_minimizers.fasta");
        std::uint64_t minimizer_size = 3;
        std::uint64_t window_size = 2;
        std::uint64_t number_of_reads = 2;

        std::vector<std::string> read_id_to_read_name;
        read_id_to_read_name.push_back("read_0");
        read_id_to_read_name.push_back("read_1");

        std::vector<std::uint32_t> read_id_to_read_length;
        read_id_to_read_length.push_back(7);
        read_id_to_read_length.push_back(6);

        std::map<representation_t, std::vector<Minimizer>> representation_to_minimizers;
        { // 032
            std::vector<Minimizer> minimizers{{0b001110, 0, SketchElement::DirectionOfRepresentation::REVERSE, 0}};
            representation_to_minimizers[0b001110] = minimizers;
        }
        { // 031
            std::vector<Minimizer> minimizers{{0b001101, 1, SketchElement::DirectionOfRepresentation::FORWARD, 0}};
            representation_to_minimizers[0b001101] = minimizers;
        }
        { // 100
            std::vector<Minimizer> minimizers{{0b010000, 3, SketchElement::DirectionOfRepresentation::FORWARD, 0}};
            representation_to_minimizers[0b010000] = minimizers;
        }
        { // 002
            std::vector<Minimizer> minimizers{{0b000010, 4, SketchElement::DirectionOfRepresentation::FORWARD, 0},
                                              {0b000010, 0, SketchElement::DirectionOfRepresentation::FORWARD, 1}};
            representation_to_minimizers[0b000010] = minimizers;
        }
        { // 021
            std::vector<Minimizer> minimizers{{0b001001, 2, SketchElement::DirectionOfRepresentation::REVERSE, 1}};
            representation_to_minimizers[0b001001] = minimizers;
        }
        { // 130
            std::vector<Minimizer> minimizers{{0b011100, 3, SketchElement::DirectionOfRepresentation::FORWARD, 1}};
            representation_to_minimizers[0b011100] = minimizers;
        }

        test_function(filename,
                      minimizer_size,
                      window_size,
                      number_of_reads,
                      read_id_to_read_name,
                      read_id_to_read_length,
                      representation_to_minimizers);

    }

}
