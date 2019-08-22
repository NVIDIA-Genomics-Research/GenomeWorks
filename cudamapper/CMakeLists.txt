#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

project(cudamapper)

# Process data subdirectory first
add_subdirectory(data)

find_package(CUDA 9.0 QUIET REQUIRED)

if(NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
else()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -use_fast_math -Xcompiler -Wall,-Wno-pedantic -std=c++14")
endif()

add_library(index_generator_cpu
        src/index_generator_cpu.cpp
        src/minimizer.cpp)
target_include_directories(index_generator_cpu PUBLIC include)
target_link_libraries(index_generator_cpu bioparser logging pthread)
target_compile_options(index_generator_cpu PRIVATE -Werror)

cuda_add_library(index_generator_gpu
        src/index_generator_gpu.cu
        src/minimizer.cpp)
target_include_directories(index_generator_gpu PUBLIC include)
target_link_libraries(index_generator_gpu bioparser logging pthread utils)
target_compile_options(index_generator_gpu PRIVATE -Werror)

add_library(index_cpu
        src/index_cpu.cpp)
target_include_directories(index_cpu PUBLIC include)
target_link_libraries(index_cpu bioparser logging pthread utils)
target_compile_options(index_cpu PRIVATE -Werror)

cuda_add_library(matcher
        src/matcher.cu)
target_include_directories(matcher PUBLIC include)
target_link_libraries(matcher utils)
target_compile_options(matcher PRIVATE -Werror)

add_library(overlapper_naive
        src/overlapper_naive.cpp)
target_include_directories(overlapper_naive PUBLIC include)
target_compile_options(overlapper_naive PRIVATE -Werror)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/cudamapper)

cuda_add_executable(cudamapper
        src/cudamapper.cpp
        src/main.cpp
        src/sequence.cpp
        src/bioparser_sequence.cpp
        src/index.cpp
        src/index_generator.cpp
        src/overlapper.cpp
        src/overlapper_naive.cpp
        )

target_compile_options(cudamapper PRIVATE -Werror)

target_include_directories(cudamapper
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cudamapper bioparser utils index_generator_cpu index_generator_gpu index_cpu matcher logging)

# Add tests folder
add_subdirectory(tests)

install(TARGETS cudamapper 
    EXPORT cudamapper
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT cudamapper DESTINATION cmake)
