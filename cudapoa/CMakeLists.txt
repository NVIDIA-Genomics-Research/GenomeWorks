#
# Copyright 2019-2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#



set(MODULE_NAME cudapoa)

# Process data subdirectory first
add_subdirectory(data)

GitVersion()

configure_file(${PROJECT_SOURCE_DIR}/common/base/src/version.cpp.in
                ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

# Optimization flags interfere with nvcc and cause compilation issues with reworked
# file structure.
if(NOT DEFINED gw_cuda_before_10_0)
    message(FATAL_ERROR "${MODULE_NAME} : variable gw_cuda_before_10_0 is not defined yet. Please make sure CUDA.cmake is loaded first.")
elseif(gw_cuda_before_10_0)
    message(STATUS "${MODULE_NAME} : Remove -O optimization when building for CUDA < 10 as it causes compilation issues.")
    string(REGEX REPLACE "-O[0-3]" "" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
    string(REGEX REPLACE "-O[0-3]" "" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
endif()

# Project specific NVCC flags, --expt-relaxed-constexpr is being added to allow using numeric_limits inside device kernels.
# Using in CUDA 10.1+ to avoid compilation issues with CUDA <= 10.0.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
if(gw_cuda_after_10_0)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
endif()
message(STATUS "nvcc flags for ${MODULE_NAME}: ${CUDA_NVCC_FLAGS}")

get_property(gw_library_type GLOBAL PROPERTY gw_library_type)
cuda_add_library(${MODULE_NAME} ${gw_library_type}
    src/cudapoa.cpp
    src/batch.cu
    src/utils.cu
    ${CMAKE_CURRENT_BINARY_DIR}/version.cpp
    )

target_link_libraries(${MODULE_NAME} gwbase gwio)

target_compile_options(${MODULE_NAME} PRIVATE -Werror -Wall -Wextra)
if (gw_optimize_for_native_cpu)
    target_compile_options(${MODULE_NAME} PRIVATE -march=native)
endif()

if(gw_enable_cudapoa_nw_print)
    target_compile_definitions(${MODULE_NAME} PUBLIC NW_VERBOSE_PRINT)
endif()

if(spoa_accurate)
    target_compile_definitions(${MODULE_NAME} PUBLIC SPOA_ACCURATE)
endif()

target_include_directories(${MODULE_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(${MODULE_NAME}-bin
        src/main.cpp
        src/application_parameters.hpp
        src/application_parameters.cpp
)

target_include_directories(${MODULE_NAME}-bin
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(${MODULE_NAME}-bin PRIVATE -Werror)
target_link_libraries(${MODULE_NAME}-bin gwio ${MODULE_NAME} gwbase)
set_target_properties(${MODULE_NAME}-bin PROPERTIES OUTPUT_NAME ${MODULE_NAME})

install(TARGETS ${MODULE_NAME}
    COMPONENT gwlogging
    EXPORT ${MODULE_NAME}
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${MODULE_NAME} DESTINATION cmake)

install(TARGETS ${MODULE_NAME}-bin
    EXPORT ${MODULE_NAME}-bin
    DESTINATION bin
)

# Add other folders.
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(samples)


# Add auto formatting.
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
