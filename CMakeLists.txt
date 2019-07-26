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
project(ClaraGenomicsAnalysisCPP)

# Process options.
option(cga_enable_tests "Build ClaraGenomicsAnalysis unit tests" OFF)
option(cga_enable_benchmarks "Build ClaraGenomicsAnalysis benchmarks" OFF)
option(cga_build_shared "Build ClaraGenomicsAnalysis libraries as shared objects" OFF)
# The spoa_accurate option runs a different (and slower) version of
# the topological sort in cudapoa which exactly matches the output
# of the topological sort implementation in the SPOA library. This is
# useful for validation. When the option is turned off, a faster
# custom top sort function is run which outputs a correct but different
# ordering of the graph. This leads to subtle differences in the
# overall consensus generated, and hence makes it harder to validate and debug.
option(spoa_accurate "Run cudapoa code in mode that matches spoa" OFF)
option(cga_enable_cudapoa_nw_print "Enable verbose prints within cudapoa NW kernel" OFF)

if (cga_enable_tests)
    message(STATUS "Enabling ClaraGenomicsAnalysis unit tests")
    set_property(GLOBAL PROPERTY enable_tests ON)
endif()

if (cga_enable_benchmarks)
    message(STATUS "Enabling ClaraGenomicsAnalysis benchmarks")
    set_property(GLOBAL PROPERTY enable_benchmarks ON)
endif()

if (cga_build_shared)
    message(STATUS "Building ClaraGenomicsAnalysis libraries as shared objects")
    set_property(GLOBAL PROPERTY cga_library_type SHARED)
else()
    message(STATUS "Building ClaraGenomicsAnalysis libraries as static objects")
    set_property(GLOBAL PROPERTY cga_library_type STATIC)
endif()

include(cmake/CUDA.cmake)
include(cmake/Doxygen.cmake)
include(cmake/3rdparty.cmake)
include(cmake/Tests.cmake)
include(cmake/Benchmarks.cmake)
include(cmake/Format.cmake)

set(CMAKE_CXX_STANDARD 17)

# Add ClaraGenomicsAnalysis projects.
add_subdirectory(common/logging)
add_subdirectory(common/cgautils)
add_subdirectory(cudapoa)
add_subdirectory(cudaaligner)

# Add documentation generation.
set_doxygen_mainpage(${CMAKE_CURRENT_SOURCE_DIR}/README.md)
add_docs_target("ClaraGenomicsAnalysis" "0.1.0")

# Add auto formatting.
cga_enable_formatting_targets()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "The default SDK install path is install" FORCE)
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
