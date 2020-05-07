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
    # Determine arch flags
    validate_boolean(cga_cuda_gen_all_arch)
    if (cga_cuda_gen_all_arch)
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Common")
    else()
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
    endif()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")
    message(STATUS "nvcc flags for cudamapper: ${CUDA_NVCC_FLAGS}")
endif()

cuda_add_library(cudamapper
        src/cudamapper.cpp
        src/index_batcher.cu
        src/index_descriptor.cpp
        src/index.cu
        src/index_cache.cu
        src/index_gpu.cu
        src/index_host_copy.cu
        src/minimizer.cu
        src/matcher.cu
        src/matcher_gpu.cu
        src/cudamapper_utils.cpp
        src/overlapper.cpp
        src/overlapper_triggered.cu)

target_include_directories(cudamapper
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cudamapper cgabase cgaio cub)
target_compile_options(cudamapper PRIVATE -Werror)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

cuda_add_executable(cudamapper-bin
        src/main.cu
)

target_compile_options(cudamapper-bin PRIVATE -Werror)
target_link_libraries(cudamapper-bin cudamapper cudaaligner)
set_target_properties(cudamapper-bin PROPERTIES OUTPUT_NAME cudamapper)


# Add tests folder
add_subdirectory(tests)

install(TARGETS cudamapper
    EXPORT cudamapper
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT cudamapper DESTINATION cmake)

install(TARGETS cudamapper-bin
    EXPORT cudamapper-bin
    DESTINATION bin
)

# Add auto formatting.
cga_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
