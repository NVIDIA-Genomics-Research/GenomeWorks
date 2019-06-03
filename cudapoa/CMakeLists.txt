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
project(cudapoa)

find_package(CUDA 9.0 QUIET REQUIRED)

if(NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
else()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo -use_fast_math -Xcompiler -Wall,-Wno-pedantic")
endif()

cuda_add_library(cudapoa STATIC 
    src/cudapoa.cpp
    src/cudapoa_kernels.cu
    src/cudapoa_topsort.cu
    src/cudapoa_add_alignment.cu
    src/cudapoa_nw.cu
    src/cudapoa_nw_banded.cu
    src/batch.cpp
    src/cudapoa_batch.cpp
    src/allocate_block.cpp
    )

target_link_libraries(cudapoa cudautils logging utils)

target_compile_options(cudapoa PRIVATE -Werror -Wall -Wextra -march=native)

if(gw_enable_cudapoa_nw_print)
    target_compile_definitions(cudapoa PUBLIC NW_VERBOSE_PRINT)
endif()

if(spoa_accurate)
    target_compile_definitions(cudapoa PUBLIC SPOA_ACCURATE)
endif()

target_include_directories(cudapoa 
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/cudapoa)

# Add tests folder
add_subdirectory(tests)

install(TARGETS cudapoa
    COMPONENT logging
    EXPORT cudapoa
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT cudapoa DESTINATION cmake)

# Add auto formatting.
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
