#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

set(TARGET_NAME cudapoatests)

set(SOURCES
    main.cpp
    Test_CudapoaBatch.cu
    Test_CudapoaTopSort.cu
    Test_CudapoaAddAlignment.cu
    Test_CudapoaNW.cu
    Test_CudapoaGenerateConsensus.cu
    Test_CudapoaBatchEnd2End.cu
    Test_CudapoaGenerateMSA2.cu
    Test_CudapoaSerializeGraph.cpp)

get_property(cudapoa_data_include_dir GLOBAL PROPERTY cudapoa_data_include_dir)
include_directories(${cudapoa_data_include_dir})

set(LIBS
    cudapoa
    spoa)

cga_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
