#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Set CXX Standard.
set(CMAKE_CXX_STANDARD 17)

#Add OpenMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# Add -O2 optimization to debug builds to speed up runtime.
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")

if(NOT DEFINED cga_cuda_before_10_0)
    message(FATAL_ERROR "variable cga_cuda_before_10 is not defined yet. Please make sure CUDA.cmake is loaded first.")
elseif(cga_cuda_before_10_0)
    message(STATUS "Remove -O optimization when building for CUDA < 10 as it causes compilation issues.")
    string(REGEX REPLACE "-O[0-3]" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REGEX REPLACE "-O[0-3]" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()
