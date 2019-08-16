#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

cmake_minimum_required(VERSION 3.10.2)
project(cudamappertests)

set(SOURCES
    main.cpp
    Test_CudamapperIndexCPU.cpp
    Test_CudamapperIndexGeneratorCPU.cpp
    Test_CudamapperIndexGeneratorGPU.cpp
    Test_CudamapperMatcher.cpp
    Test_PAF_generator.cpp
    ../src/bioparser_sequence.cpp)

get_property(cudamapper_data_include_dir GLOBAL PROPERTY cudamapper_data_include_dir)
include_directories(${cudamapper_data_include_dir})

set(LIBS
    bioparser
    index_generator_cpu
    index_generator_gpu
    index_cpu
    matcher
    overlapper_naive)

cga_add_tests(${PROJECT_NAME} "${SOURCES}" "${LIBS}")
