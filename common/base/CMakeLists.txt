#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

project(gwbase)

find_package(CUDA 9.0 QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
message(STATUS "nvcc flags for ${PROJECT_NAME}: ${CUDA_NVCC_FLAGS}")

get_property(gw_library_type GLOBAL PROPERTY gw_library_type)
add_library(${PROJECT_NAME} ${gw_library_type}
        src/cudautils.cpp
        src/logging.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog ${CUDA_LIBRARIES})

if (gw_profiling)
    find_library(NVTX_LIBRARY nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DGW_PROFILING)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${NVTX_LIBRARY})
endif()

if (gw_device_synchronize_kernels)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GW_DEVICE_SYNCHRONIZE)
endif()

if(gw_enable_caching_allocator)
    target_compile_definitions(${PROJECT_NAME} PUBLIC GW_ENABLE_CACHING_ALLOCATOR)
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CUDA_INCLUDE_DIRS}
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${PROJECT_NAME} DESTINATION cmake)

# Add documentation
add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add tests
add_subdirectory(tests)

# Adding formatting
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
