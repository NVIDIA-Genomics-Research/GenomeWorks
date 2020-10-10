#
# Copyright 2020 NVIDIA CORPORATION.
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
set(MODULE_NAME cudaextender)

# Process data subdirectory first
add_subdirectory(data)

GitVersion()

configure_file(${PROJECT_SOURCE_DIR}/common/base/src/version.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

# Project specific NVCC flags, --expt-relaxed-constexpr is being added to allow using numeric_limits inside device kernels.
# Using in CUDA 10.1+ to avoid compilation issues with CUDA <= 10.0.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
if (gw_cuda_after_10_0)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
endif ()
if (gw_cuda_gen_all_arch)
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Common")
else ()
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
endif ()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")
message(STATUS "nvcc flags for ${PROJECT_NAME}: ${CUDA_NVCC_FLAGS}")

get_property(gw_library_type GLOBAL PROPERTY gw_library_type)
cuda_add_library(${MODULE_NAME} ${gw_library_type}
        src/cudaextender.cpp
        src/extender.cpp
        src/ungapped_xdrop.cu
        src/ungapped_xdrop_kernels.cu
        ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

target_link_libraries(${MODULE_NAME} gwbase gwio cub)

target_compile_options(${MODULE_NAME} PRIVATE -Werror -Wall -Wextra)
if (gw_optimize_for_native_cpu)
    target_compile_options(${MODULE_NAME} PRIVATE -march=native)
endif ()

target_include_directories(${MODULE_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        )

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS ${MODULE_NAME}
        COMPONENT gwlogging
        EXPORT ${MODULE_NAME}
        DESTINATION lib
        INCLUDES DESTINATION include
        )
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${MODULE_NAME} DESTINATION cmake)

# Add other folders.
add_subdirectory(samples)
add_subdirectory(tests)

# Add auto formatting.
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
