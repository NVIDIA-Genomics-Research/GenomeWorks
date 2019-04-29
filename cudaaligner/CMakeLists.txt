cmake_minimum_required(VERSION 3.10.2)
project(cudaaligner)

find_package(CUDA 9.0 QUIET REQUIRED)

if(NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
else()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -use_fast_math -Xcompiler -Wno-pedantic")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native")

cuda_add_library(cudaaligner STATIC 
    src/cudaaligner.cpp
    )

target_include_directories(cudaaligner 
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/cudaaligner)

# Add tests folder
add_subdirectory(tests)

install(TARGETS cudaaligner
    EXPORT cudaaligner
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT cudaaligner DESTINATION cmake)
