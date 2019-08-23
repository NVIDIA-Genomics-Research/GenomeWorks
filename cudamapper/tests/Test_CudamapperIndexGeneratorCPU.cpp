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
#include "../src/index_generator_cpu.hpp"

namespace claragenomics {

    TEST(TestCudamapperIndexGeneratorCPU, OneReadOneMinimizer) {
        // >read_0
        // GATT

        IndexGeneratorCPU index_generator(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/one_read_one_minimizer.fasta", 4, 1);
        EXPECT_EQ(index_generator.minimizer_size(), 4);
        EXPECT_EQ(index_generator.window_size(), 1);
        ASSERT_EQ(index_generator.number_of_reads(), 1);

        ASSERT_EQ(index_generator.read_id_to_read_name().size(), 1);
        EXPECT_EQ(index_generator.read_id_to_read_name()[0], std::string("read_0"));

        const auto& representations_and_sketch_elements = index_generator.representations_and_sketch_elements();
        ASSERT_EQ(representations_and_sketch_elements.size(), 1);
        ASSERT_EQ(representations_and_sketch_elements[0].representation_, 0b00001101);

        const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = representations_and_sketch_elements[0].sketch_elements_;
        ASSERT_EQ(sketch_elements_for_representation.size(), 1);
        const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
        EXPECT_EQ(minimizer.representation(), 0b00001101);
        EXPECT_EQ(minimizer.position_in_read(), 0);
        EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::REVERSE);
        EXPECT_EQ(minimizer.read_id(), 0);
    }

    TEST(TestCudaMapperIndexGeneratorCPU, TwoReadsMultipleMinimizers) {
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

        // complete datastructure: AAG(4f0), AAG(0f1), AGC(1f1), AGC(2r1), ATC(1f0), ATG(0r0), CAA(3f0), CTA(3f1)

        IndexGeneratorCPU index_generator(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/two_reads_multiple_minimizers.fasta", 3, 2);
        EXPECT_EQ(index_generator.minimizer_size(), 3);
        EXPECT_EQ(index_generator.window_size(), 2);
        ASSERT_EQ(index_generator.number_of_reads(), 2);

        ASSERT_EQ(index_generator.read_id_to_read_name().size(), 2);
        EXPECT_EQ(index_generator.read_id_to_read_name()[0], std::string("read_0"));
        EXPECT_EQ(index_generator.read_id_to_read_name()[1], std::string("read_1"));

        const auto& representations_and_sketch_elements = index_generator.representations_and_sketch_elements();
        ASSERT_EQ(representations_and_sketch_elements.size(), 6);

        // complete datastructure: AAG(4f0), AAG(0f1), AGC(1f1), AGC(2r1), ATC(1f0), ATG(0r0), CAA(3f0), CTA(3f1)
        // AAG (4f0), (0f1)
        {
            ASSERT_EQ(representations_and_sketch_elements[0].representation_, 0b000010) << "AAG";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = representations_and_sketch_elements[0].sketch_elements_;
            ASSERT_EQ(sketch_elements_for_representation.size(), 2) << "AAG";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0b000010) << "AAG (4f0)";
                EXPECT_EQ(minimizer.position_in_read(), 4) << "AAG (4f0)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "AAG (4f0)";
                EXPECT_EQ(minimizer.read_id(), 0) << "AAG (4f0)";
            }
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[1]);
                EXPECT_EQ(minimizer.representation(), 0b000010) << "AAG (0f1)";
                EXPECT_EQ(minimizer.position_in_read(), 0) << "AAG (0f1)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "AAG (0f1)";
                EXPECT_EQ(minimizer.read_id(), 1) << "AAG (0f1)";
            }
        }
        // AGC (1f1), (2r1)
        {
            ASSERT_EQ(representations_and_sketch_elements[1].representation_, 0b001001) << "AGC";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = representations_and_sketch_elements[1].sketch_elements_;
            ASSERT_EQ(sketch_elements_for_representation.size(), 2) << "AGC";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0b001001) << "AGC (1f1)";
                EXPECT_EQ(minimizer.position_in_read(), 1) << "AGC (1f1)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "AGC (1f1)";
                EXPECT_EQ(minimizer.read_id(), 1) << "AGC (1f1)";
            }
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[1]);
                EXPECT_EQ(minimizer.representation(), 0b001001) << "AGC (2r1)";
                EXPECT_EQ(minimizer.position_in_read(), 2) << "AGC (r21)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::REVERSE) << "AGC (2r1)";
                EXPECT_EQ(minimizer.read_id(), 1) << "AGC (2r1)";
            }
        }
        // ATC (1f0)
        {
            ASSERT_EQ(representations_and_sketch_elements[2].representation_, 0b001101) << "ATC";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = representations_and_sketch_elements[2].sketch_elements_;
            ASSERT_EQ(sketch_elements_for_representation.size(), 1) << "ATC";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0b001101) << "ATC (1f0)";
                EXPECT_EQ(minimizer.position_in_read(), 1) << "ATC (1f0)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "ATC (1f0)";
                EXPECT_EQ(minimizer.read_id(), 0) << "ATC (1f0)";
            }
        }
        // ATG (0r0)
        {
            ASSERT_EQ(representations_and_sketch_elements[3].representation_, 0b001110) << "ATG";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = representations_and_sketch_elements[3].sketch_elements_;
            ASSERT_EQ(sketch_elements_for_representation.size(), 1) << "ATG";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0b001110) << "ATG (0r0)";
                EXPECT_EQ(minimizer.position_in_read(), 0) << "ATG (0r0)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::REVERSE) << "ATG (0r0)";
                EXPECT_EQ(minimizer.read_id(), 0) << "ATG (0r0)";
            }
        }
        // CAA (3f0)
        {
            ASSERT_EQ(representations_and_sketch_elements[4].representation_, 0b010000) << "CAA";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = representations_and_sketch_elements[4].sketch_elements_;
            ASSERT_EQ(sketch_elements_for_representation.size(), 1) << "CAA";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0b010000) << "CAA (3f0)";
                EXPECT_EQ(minimizer.position_in_read(), 3) << "CAA (3f0)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "CAA (3f0)";
                EXPECT_EQ(minimizer.read_id(), 0) << "CAA (3f0)";
            }
        }
        // CTA (3f1)
        {
            ASSERT_EQ(representations_and_sketch_elements[5].representation_, 0b011100) << "CTA";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = representations_and_sketch_elements[5].sketch_elements_;
            ASSERT_EQ(sketch_elements_for_representation.size(), 1) << "CTA";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0b011100) << "CTA (3f1)";
                EXPECT_EQ(minimizer.position_in_read(), 3) << "CTA (3f1)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "CTA (3f1)";
                EXPECT_EQ(minimizer.read_id(), 1) << "CTA (3f1)";
            }
        }
    }
}
