#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Check CUDA dependency for project.
find_package(CUDA 9.0 REQUIRED)

if(NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
else()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -use_fast_math -Xcompiler -Wall,-Wno-pedantic")
    # Determine arch flags
    validate_boolean(cga_cuda_gen_all_arch)
    if (cga_cuda_gen_all_arch)
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Common")
    else()
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
    endif()
    message(STATUS "NVCC arch flags - ${ARCH_FLAGS}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")
endif()

