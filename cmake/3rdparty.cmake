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

# Add 3rd party build dependencies.
if (NOT TARGET bioparser)
    add_subdirectory(3rdparty/bioparser EXCLUDE_FROM_ALL)
endif()

get_property(enable_tests GLOBAL PROPERTY enable_tests)
if (enable_tests AND NOT TARGET gtest)
    add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
endif()

if (NOT TARGET spdlog)
# FORCE spdlog to put out an install target, which we need
    set(SPDLOG_INSTALL ON CACHE BOOL "Generate the install target." FORCE)
    add_subdirectory(3rdparty/spdlog EXCLUDE_FROM_ALL)
endif()

