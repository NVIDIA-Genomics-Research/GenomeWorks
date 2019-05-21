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
project(cudapoatests)

set(SOURCES
    main.cpp
    Test_CudapoaBatch.cpp
    Test_CudapoaTopSort.cpp
    Test_CudapoaAddAlignment.cpp
    Test_CudapoaNW.cpp)

set(LIBS
    cudapoa)

gw_add_tests(${PROJECT_NAME} "${SOURCES}" "${LIBS}")
