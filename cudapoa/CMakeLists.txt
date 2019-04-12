cmake_minimum_required(VERSION 3.10.2)
project(cudapoa)

find_package(CUDA 9.0 QUIET REQUIRED)

if(NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
else()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -use_fast_math -Xcompiler -Wno-pedantic")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")

cuda_add_library(cudapoa STATIC 
    src/cudapoa_kernels.cu
    src/cudapoa_topsort.cu
    src/cudapoa_add_alignment.cu
    src/cudapoa_nw.cu
    src/cudapoa_batch.cpp
    )

if(debug)
    target_compile_definitions(cudapoa PUBLIC DEBUG)
endif()

if(spoa_accurate)
    target_compile_definitions(cudapoa PUBLIC SPOA_ACCURATE)
endif()

target_include_directories(cudapoa PUBLIC include)


# Add tests folder
add_subdirectory(tests)

install(TARGETS cudapoa DESTINATION lib)
install(FILES include/cudapoa/cudapoa_batch.hpp DESTINATION include/cudapoa)