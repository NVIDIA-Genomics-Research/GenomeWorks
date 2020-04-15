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
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo --expt-extended-lambda -use_fast_math -Xcompiler -Wall,-Wno-pedantic -std=c++14")
endif()

cuda_add_library(minimizer
        src/minimizer.cu)
target_include_directories(minimizer PUBLIC include ${CUB_DIR})
target_link_libraries(minimizer cgabase cgalogging pthread)
target_compile_options(minimizer PRIVATE -Werror)

cuda_add_library(index_gpu
        src/index.cu
        src/index_gpu.cu
        src/index_host_copy.cu
        src/minimizer.cu)
target_include_directories(index_gpu PUBLIC include ${CUB_DIR})
target_link_libraries(index_gpu cgabase cgaio cgalogging minimizer pthread)
target_compile_options(index_gpu PRIVATE -Werror)

cuda_add_library(matcher_gpu
        src/matcher_gpu.cu)
target_include_directories(matcher_gpu PUBLIC include ${CUB_DIR})
target_link_libraries(matcher_gpu cgabase cgaio cgalogging)
target_compile_options(matcher_gpu PRIVATE -Werror)

add_library(cudamapper_utils
        src/cudamapper_utils.cpp)
target_include_directories(cudamapper_utils PUBLIC include)
target_link_libraries(cudamapper_utils cgabase)

cuda_add_library(overlapper_triggerred
        src/overlapper_triggered.cu
        src/overlapper.cpp)
target_include_directories(overlapper_triggerred PUBLIC include ${CUB_DIR})
target_link_libraries(overlapper_triggerred cgabase cgaio cgalogging cudaaligner)
target_compile_options(overlapper_triggerred PRIVATE -Werror)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

cuda_add_executable(cudamapper
        src/cudamapper.cpp
        src/index_cache.cu
        src/index_descriptor.cpp
        src/main.cu
        src/matcher.cu
        src/overlapper.cpp
)

if(cga_enable_caching_allocator)
        if (cga_enable_preallocating_caching_allocator)
                target_compile_definitions(cudamapper            PUBLIC CGA_ENABLE_CACHING_ALLOCATOR CGA_ENABLE_PREALLOCATING_ALLOCATOR)
                target_compile_definitions(minimizer             PUBLIC CGA_ENABLE_CACHING_ALLOCATOR CGA_ENABLE_PREALLOCATING_ALLOCATOR)
                target_compile_definitions(index_gpu             PUBLIC CGA_ENABLE_CACHING_ALLOCATOR CGA_ENABLE_PREALLOCATING_ALLOCATOR)
                target_compile_definitions(matcher_gpu           PUBLIC CGA_ENABLE_CACHING_ALLOCATOR CGA_ENABLE_PREALLOCATING_ALLOCATOR)
                target_compile_definitions(overlapper_triggerred PUBLIC CGA_ENABLE_CACHING_ALLOCATOR CGA_ENABLE_PREALLOCATING_ALLOCATOR)
        else()
                target_compile_definitions(cudamapper            PUBLIC CGA_ENABLE_CACHING_ALLOCATOR)
                target_compile_definitions(minimizer             PUBLIC CGA_ENABLE_CACHING_ALLOCATOR)
                target_compile_definitions(index_gpu             PUBLIC CGA_ENABLE_CACHING_ALLOCATOR)
                target_compile_definitions(matcher_gpu           PUBLIC CGA_ENABLE_CACHING_ALLOCATOR)
                target_compile_definitions(overlapper_triggerred PUBLIC CGA_ENABLE_CACHING_ALLOCATOR)
        endif()
endif()

target_compile_options(cudamapper PRIVATE -Werror)

target_include_directories(cudamapper
    PRIVATE
        ${THREADPOOL_DIR}
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(cudamapper cgabase cgalogging cudamapper_utils index_gpu matcher_gpu overlapper_triggerred)

# Add tests folder
add_subdirectory(tests)

install(TARGETS cudamapper 
    EXPORT cudamapper
    DESTINATION bin
)
install(EXPORT cudamapper DESTINATION cmake)

# Add auto formatting.
cga_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
