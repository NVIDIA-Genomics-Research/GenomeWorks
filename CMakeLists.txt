#
# Copyright 2019-2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#



cmake_minimum_required(VERSION 3.10.2)
set(GW_PROJECT_NAME GenomeWorks)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" GW_VERSION)
string(STRIP ${GW_VERSION} GW_VERSION)
project(${GW_PROJECT_NAME})

# Process options.
option(gw_enable_tests "Build GenomeWorks unit tests" OFF)
option(gw_enable_benchmarks "Build GenomeWorks benchmarks" OFF)
option(gw_build_shared "Build GenomeWorks libraries as shared objects" OFF)
option(gw_device_synchronize_kernels "Run cudaDeviceSynchronize() in GW_CU_CHECK_ERR calls" OFF)
option(gw_optimize_for_native_cpu "Build with march=native" ON)
# The spoa_accurate option runs a different (and slower) version of
# the topological sort in cudapoa which exactly matches the output
# of the topological sort implementation in the SPOA library. This is
# useful for validation. When the option is turned off, a faster
# custom top sort function is run which outputs a correct but different
# ordering of the graph. This leads to subtle differences in the
# overall consensus generated, and hence makes it harder to validate and debug.
option(spoa_accurate "Run cudapoa code in mode that matches spoa" OFF)
option(gw_enable_cudapoa_nw_print "Enable verbose prints within cudapoa NW kernel" OFF)
option(gw_profiling "Compile a binary for profiling with NVTX markers." OFF)
option(gw_enable_caching_allocator "Enable caching allocator." ON)
option(gw_generate_docs "Generate Doxygen documentation" ON)
option(gw_cuda_gen_all_arch "ON: Generate optimized CUDA code for all architectures | OFF: for detected architectures only" OFF)

# Must be included before others for options value validation
include(cmake/Utils.cmake)

validate_boolean(gw_enable_tests)
if (gw_enable_tests)
    message(STATUS "Enabling GenomeWorks unit tests")
    set_property(GLOBAL PROPERTY enable_tests ON)
endif()

validate_boolean(gw_enable_benchmarks)
if (gw_enable_benchmarks)
    message(STATUS "Enabling GenomeWorks benchmarks")
    set_property(GLOBAL PROPERTY enable_benchmarks ON)
endif()

validate_boolean(gw_build_shared)
if (gw_build_shared)
    message(STATUS "Building GenomeWorks libraries as shared objects")
    set_property(GLOBAL PROPERTY gw_library_type SHARED)
else()
    message(STATUS "Building GenomeWorks libraries as static objects")
    set_property(GLOBAL PROPERTY gw_library_type STATIC)
endif()

include(cmake/CXX.cmake)
include(cmake/CUDA.cmake)
include(cmake/Doxygen.cmake)
include(cmake/3rdparty.cmake)
include(cmake/Tests.cmake)
include(cmake/Benchmarks.cmake)
include(cmake/Format.cmake)
include(cmake/Packaging.cmake)
include(cmake/CondaEnv.cmake)
include(cmake/GitVersion.cmake)

# Add GenomeWorks projects.
add_subdirectory(common/base)
add_subdirectory(common/io)
add_subdirectory(cudapoa)
add_subdirectory(cudamapper)
add_subdirectory(cudaaligner)

# Add documentation generation.
validate_boolean(gw_generate_docs)
if (gw_generate_docs)
    message(STATUS "Enabling Doxygen documentation generation")
    set_doxygen_mainpage(${CMAKE_CURRENT_SOURCE_DIR}/README.md)
    add_docs_target("GenomeWorks" "${GW_VERSION}")
else()
    message(STATUS "Disabling Doxygen documentation generation")
endif()

# Add auto formatting.
gw_enable_formatting_targets()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT  AND (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR}))
    set (CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "The default SDK install path is install" FORCE)
endif ()
