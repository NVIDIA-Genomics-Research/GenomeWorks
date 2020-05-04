#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

project(cgabase)

find_package(CUDA 9.0 QUIET REQUIRED)

get_property(cga_library_type GLOBAL PROPERTY cga_library_type)
add_library(${PROJECT_NAME} ${cga_library_type}
        src/cudautils.cpp
        src/logging.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC spdlog ${CUDA_LIBRARIES})

if (cga_profiling)
    find_library(NVTX_LIBRARY nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DCGA_PROFILING)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${NVTX_LIBRARY})
endif()

if (cga_device_synchronize_kernels)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CGA_DEVICE_SYNCHRONIZE)
endif()

if(cga_enable_caching_allocator)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CGA_ENABLE_CACHING_ALLOCATOR)
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
cga_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
