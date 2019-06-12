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

    TEST(TestCudamapperIndexGeneratorCPU, SingleMinimizer_k4_w1) {
        IndexGeneratorCPU index_generator(std::string("cudamapper/tests/data/single_minimizer.fasta"), 4, 1);
        ASSERT_EQ(index_generator.number_of_reads(), 1);
    }

}
