#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

set(TARGET_NAME cgadefstests)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -Xcompiler -Wall,-Wno-pedantic -std=c++14")

set(SOURCES
    main.cpp
    Test_DefsIndexDescriptor.cpp)

include_directories(${CUB_DIR})

set(LIBS
    cgadefs)

cga_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
