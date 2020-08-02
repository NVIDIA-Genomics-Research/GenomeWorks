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



# Add 3rd party build dependencies.
get_property(enable_tests GLOBAL PROPERTY enable_tests)
if (enable_tests AND NOT TARGET gtest)
    add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
endif()

get_property(enable_benchmarks GLOBAL PROPERTY enable_benchmarks)
if (enable_benchmarks AND NOT TARGET benchmark)
    set(BENCHMARK_ENABLE_TESTING OFF)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
    add_subdirectory(3rdparty/benchmark EXCLUDE_FROM_ALL)
endif()

if (NOT TARGET spdlog)
# FORCE spdlog to put out an install target, which we need
    set(SPDLOG_INSTALL ON CACHE BOOL "Generate the install target." FORCE)
    add_subdirectory(3rdparty/spdlog EXCLUDE_FROM_ALL)
endif()

if (NOT TARGET spoa)
    add_subdirectory(3rdparty/spoa EXCLUDE_FROM_ALL)
# Don't show warnings when compiling the 3rd party library
    target_compile_options(spoa PRIVATE -w)
endif()

set(CUB_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cub CACHE STRING
    "Path to cub repo")
add_library(cub INTERFACE IMPORTED)
#target_include_directories(cub INTERFACE ${CUB_DIR}>) does not work with
#cmake before 3.11, use the following for now:
set_property(TARGET cub APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUB_DIR}")

set(KSEQPP_DIR ${PROJECT_SOURCE_DIR}/3rdparty/kseqpp/src CACHE STRING
    "Path to kseqpp repo")
