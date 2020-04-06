#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

project(cgaconfig)

find_package(CUDA 9.0 QUIET REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_link_libraries(${PROJECT_NAME} INTERFACE ${CUDA_LIBRARIES})

if (cga_profiling)
    find_library(NVTX_LIBRARY nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_compile_definitions(${PROJECT_NAME} INTERFACE -DCGA_PROFILING)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${NVTX_LIBRARY})
endif()

target_include_directories(${PROJECT_NAME}
    INTERFACE 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CUDA_INCLUDE_DIRS}
)

if (cga_device_synchronize_kernels)
    target_compile_definitions(${PROJECT_NAME} INTERFACE CGA_DEVICE_SYNCHRONIZE)
endif()


install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${PROJECT_NAME} DESTINATION cmake)

# Adding formatting
cga_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
