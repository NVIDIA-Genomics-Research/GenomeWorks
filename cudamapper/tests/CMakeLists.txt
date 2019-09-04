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
    Test_CudamapperMatcher.cpp
    Test_CudamapperMinimizer.cpp
    Test_PAF_generator.cpp
    Test_CudamapperOverlapperTriggered.cpp
    ../src/bioparser_sequence.cpp)

get_property(cudamapper_data_include_dir GLOBAL PROPERTY cudamapper_data_include_dir)
include_directories(${cudamapper_data_include_dir})

set(LIBS
    bioparser
    index_cpu
    matcher
    overlapper_naive
    overlapper_triggerred
    cudamapper_utils)

cga_add_tests(${PROJECT_NAME} "${SOURCES}" "${LIBS}")