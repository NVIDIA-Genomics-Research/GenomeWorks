#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

project(cudaaligner)

# Project specific NVCC flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr")

cuda_add_library(cudaaligner STATIC 
    src/cudaaligner.cpp
    src/aligner.cpp
    src/alignment_impl.cpp
    src/aligner_global.cpp
    src/ukkonen_gpu.cu
    )

target_link_libraries(cudaaligner cudautils logging utils)

target_compile_options(cudaaligner PRIVATE -Werror -Wall -Wextra -march=native)

target_include_directories(cudaaligner 
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(cudaaligner PRIVATE -Werror)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/cudaaligner)

# Add tests folder
add_subdirectory(tests)
add_subdirectory(benchmarks)

install(TARGETS cudaaligner
    COMPONENT logging
    EXPORT cudaaligner
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT cudaaligner DESTINATION cmake)

# Add auto formatting.
cga_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
