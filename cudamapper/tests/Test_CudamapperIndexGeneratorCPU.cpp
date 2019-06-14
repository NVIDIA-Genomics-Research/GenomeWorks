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
#include "../src/index_generator_cpu.hpp"

namespace genomeworks {

    // TODO: create absolute paths at compile time in a way similar to cudapoa/data/file_location.hpp.in

    TEST(TestCudamapperIndexGeneratorCPU, OneReadOneMinimizer) {
        // >read_0
        // GATT

        IndexGeneratorCPU index_generator(std::string("cudamapper/tests/data/one_read_one_minimizer.fasta"), 4, 1);
        EXPECT_EQ(index_generator.minimizer_size(), 4);
        EXPECT_EQ(index_generator.window_size(), 1);
        ASSERT_EQ(index_generator.number_of_reads(), 1);

        ASSERT_EQ(index_generator.read_id_to_read_name().size(), 1);
        EXPECT_EQ(index_generator.read_id_to_read_name()[0], std::string("read_0"));

        const auto& representations_to_sketch_elements = index_generator.representations_to_sketch_elements();
        ASSERT_EQ(representations_to_sketch_elements.size(), 1);
        ASSERT_NE(representations_to_sketch_elements.find(0x0031), std::end(representations_to_sketch_elements));

        const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = (*representations_to_sketch_elements.find(0x0031)).second;
        ASSERT_EQ(sketch_elements_for_representation.size(), 1);
        const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
        EXPECT_EQ(minimizer.representation(), 0x0031);
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
        // All minimizers: ATC(1f), CCA(3f), AAG(4f), ATG(0r)

        // AAGCTA
        // Central minimizers:
        // AAGC: <AAG>, CTT, AGC, GCT
        // AGCT: <AGC>, GCT, GCT, <AGC>
        // GCTA: GCT, <AGC>, CTA, TAG
        // Front end minimizers: none
        // Back end miniminers: <CTA>, TAG
        // All minimizers: AAG(0f), AGC(1f), AGC(2r), CTA(3f)

        // complete datastructure: AAG(4f0), AAG(0f1), AGC(1f1), AGC(2r1), ATC(1f0), ATG(0r0), CCA(3f0), CTA(3f1)

        IndexGeneratorCPU index_generator(std::string("cudamapper/tests/data/two_reads_multiple_minimizers.fasta"), 3, 2);
        EXPECT_EQ(index_generator.minimizer_size(), 3);
        EXPECT_EQ(index_generator.window_size(), 2);
        ASSERT_EQ(index_generator.number_of_reads(), 2);

        ASSERT_EQ(index_generator.read_id_to_read_name().size(), 2);
        EXPECT_EQ(index_generator.read_id_to_read_name()[0], std::string("read_0"));
        EXPECT_EQ(index_generator.read_id_to_read_name()[1], std::string("read_1"));

        const auto& representations_to_sketch_elements = index_generator.representations_to_sketch_elements();
        ASSERT_EQ(representations_to_sketch_elements.size(), 6);

        // complete datastructure: AAG(4f0), AAG(0f1), AGC(1f1), AGC(2r1), ATC(1f0), ATG(0r0), CCA(3f0), CTA(3f1)
        // AAG (4f0), (0f1)
        {
            ASSERT_NE(representations_to_sketch_elements.find(0x002), std::end(representations_to_sketch_elements)) << "AAG";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = (*representations_to_sketch_elements.find(0x002)).second;
            ASSERT_EQ(sketch_elements_for_representation.size(), 2) << "AAG";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0x002) << "AAG (4f0)";
                EXPECT_EQ(minimizer.position_in_read(), 4) << "AAG (4f0)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "AAG (4f0)";
                EXPECT_EQ(minimizer.read_id(), 0) << "AAG (4f0)";
            }
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[1]);
                EXPECT_EQ(minimizer.representation(), 0x002) << "AAG (0f1)";
                EXPECT_EQ(minimizer.position_in_read(), 0) << "AAG (0f1)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "AAG (0f1)";
                EXPECT_EQ(minimizer.read_id(), 1) << "AAG (0f1)";
            }
        }
        // AGC (1f1), (2r1)
        {
            ASSERT_NE(representations_to_sketch_elements.find(0x021), std::end(representations_to_sketch_elements)) << "AGC";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = (*representations_to_sketch_elements.find(0x021)).second;
            ASSERT_EQ(sketch_elements_for_representation.size(), 2) << "AGC";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0x021) << "AGC (1f1)";
                EXPECT_EQ(minimizer.position_in_read(), 1) << "AGC (1f1)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "AGC (1f1)";
                EXPECT_EQ(minimizer.read_id(), 1) << "AGC (1f1)";
            }
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[1]);
                EXPECT_EQ(minimizer.representation(), 0x021) << "AGC (2r1)";
                EXPECT_EQ(minimizer.position_in_read(), 2) << "AGC (r21)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::REVERSE) << "AGC (2r1)";
                EXPECT_EQ(minimizer.read_id(), 1) << "AGC (2r1)";
            }
        }
        // ATC (1f0)
        {
            ASSERT_NE(representations_to_sketch_elements.find(0x031), std::end(representations_to_sketch_elements)) << "ATC";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = (*representations_to_sketch_elements.find(0x031)).second;
            ASSERT_EQ(sketch_elements_for_representation.size(), 1) << "ATC";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0x031) << "ATC (1f0)";
                EXPECT_EQ(minimizer.position_in_read(), 1) << "ATC (1f0)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "ATC (1f0)";
                EXPECT_EQ(minimizer.read_id(), 0) << "ATC (1f0)";
            }
        }
        // ATG (0r0)
        {
            ASSERT_NE(representations_to_sketch_elements.find(0x032), std::end(representations_to_sketch_elements)) << "ATG";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = (*representations_to_sketch_elements.find(0x032)).second;
            ASSERT_EQ(sketch_elements_for_representation.size(), 1) << "ATG";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0x032) << "ATG (0r0)";
                EXPECT_EQ(minimizer.position_in_read(), 0) << "ATG (0r0)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::REVERSE) << "ATG (0r0)";
                EXPECT_EQ(minimizer.read_id(), 0) << "ATG (0r0)";
            }
        }
        // CAA (3f0)
        {
            ASSERT_NE(representations_to_sketch_elements.find(0x100), std::end(representations_to_sketch_elements)) << "CAA";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = (*representations_to_sketch_elements.find(0x100)).second;
            ASSERT_EQ(sketch_elements_for_representation.size(), 1) << "CAA";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0x100) << "CAA (3f0)";
                EXPECT_EQ(minimizer.position_in_read(), 3) << "CAA (3f0)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "CAA (3f0)";
                EXPECT_EQ(minimizer.read_id(), 0) << "CAA (3f0)";
            }
        }
        // CTA (3f1)
        {
            ASSERT_NE(representations_to_sketch_elements.find(0x130), std::end(representations_to_sketch_elements)) << "CTA";
            const std::vector<std::unique_ptr<SketchElement>>& sketch_elements_for_representation = (*representations_to_sketch_elements.find(0x130)).second;
            ASSERT_EQ(sketch_elements_for_representation.size(), 1) << "CTA";
            {
                const Minimizer& minimizer = static_cast<const Minimizer&>(*sketch_elements_for_representation[0]);
                EXPECT_EQ(minimizer.representation(), 0x130) << "CTA (3f1)";
                EXPECT_EQ(minimizer.position_in_read(), 3) << "CTA (3f1)";
                EXPECT_EQ(minimizer.direction(), SketchElement::DirectionOfRepresentation::FORWARD) << "CTA (3f1)";
                EXPECT_EQ(minimizer.read_id(), 1) << "CTA (3f1)";
            }
        }
    }

}
