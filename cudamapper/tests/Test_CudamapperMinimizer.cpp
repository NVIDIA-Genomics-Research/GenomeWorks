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
#include "../src/minimizer.hpp"

namespace claragenomics {
    TEST(TestCudamapperMinimizer, GATT_4) {
        const auto minimizer = Minimizer::kmer_to_representation(std::string("GATT"), 0, 4);
        EXPECT_EQ(minimizer.representation_, 0b00001101);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::REVERSE);
    }

    TEST(TestCudamapperMinimizer, GATT_2) {
        auto minimizer = Minimizer::kmer_to_representation(std::string("GATT"), 0, 2);
        EXPECT_EQ(minimizer.representation_, 0b1000);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);

        minimizer = Minimizer::kmer_to_representation(std::string("GATT"), 1, 2);
        EXPECT_EQ(minimizer.representation_, 0b0011);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);

        minimizer = Minimizer::kmer_to_representation(std::string("GATT"), 2, 2);
        EXPECT_EQ(minimizer.representation_, 0b0000);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::REVERSE);
    }

    TEST(TestCudamapperMinimizer, CCCATAC_3) {
        auto minimizer = Minimizer::kmer_to_representation(std::string("CCCATAC"), 0, 3);
        EXPECT_EQ(minimizer.representation_, 0b010101);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);

        minimizer = Minimizer::kmer_to_representation(std::string("CCCATAC"), 1, 3);
        EXPECT_EQ(minimizer.representation_, 0b010100);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);


        minimizer = Minimizer::kmer_to_representation(std::string("CCCATAC"), 2, 3);
        EXPECT_EQ(minimizer.representation_, 0b001110);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::REVERSE);


        minimizer = Minimizer::kmer_to_representation(std::string("CCCATAC"), 3, 3);
        EXPECT_EQ(minimizer.representation_, 0b001100);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);


        minimizer = Minimizer::kmer_to_representation(std::string("CCCATAC"), 4, 3);
        EXPECT_EQ(minimizer.representation_, 0b101100);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::REVERSE);
    }

    TEST(TestCudamapperMinimizer, CATCAAG_3) {
        auto minimizer = Minimizer::kmer_to_representation(std::string("CATCAAG"), 0, 3);
        EXPECT_EQ(minimizer.representation_, 0b001110);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::REVERSE);

        minimizer = Minimizer::kmer_to_representation(std::string("CATCAAG"), 1, 3);
        EXPECT_EQ(minimizer.representation_, 0b001101);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);

        minimizer = Minimizer::kmer_to_representation(std::string("CATCAAG"), 2, 3);
        EXPECT_EQ(minimizer.representation_, 0b110100);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);

        minimizer = Minimizer::kmer_to_representation(std::string("CATCAAG"), 3, 3);
        EXPECT_EQ(minimizer.representation_, 0b010000);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);

        minimizer = Minimizer::kmer_to_representation(std::string("CATCAAG"), 4, 3);
        EXPECT_EQ(minimizer.representation_, 0b000010);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);
    }

    TEST(TestCudamapperMinimizer, AAGCTA_3) {
        auto minimizer = Minimizer::kmer_to_representation(std::string("AAGCTA"), 0, 3);
        EXPECT_EQ(minimizer.representation_, 0b000010);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);

        minimizer = Minimizer::kmer_to_representation(std::string("AAGCTA"), 1, 3);
        EXPECT_EQ(minimizer.representation_, 0b001001);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);

        minimizer = Minimizer::kmer_to_representation(std::string("AAGCTA"), 2, 3);
        EXPECT_EQ(minimizer.representation_, 0b001001);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::REVERSE);

        minimizer = Minimizer::kmer_to_representation(std::string("AAGCTA"), 3, 3);
        EXPECT_EQ(minimizer.representation_, 0b011100);
        EXPECT_EQ(minimizer.direction_, SketchElement::DirectionOfRepresentation::FORWARD);
    }
}