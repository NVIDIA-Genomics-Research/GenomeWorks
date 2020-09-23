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



set(MODULE_NAME gwbase)

find_package(CUDA 9.0 QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")
message(STATUS "nvcc flags for ${MODULE_NAME}: ${CUDA_NVCC_FLAGS}")

get_property(gw_library_type GLOBAL PROPERTY gw_library_type)
add_library(${MODULE_NAME} ${gw_library_type}
        src/cudautils.cpp
        src/logging.cpp
        src/graph.cpp
        )
target_link_libraries(${MODULE_NAME} PUBLIC spdlog ${CUDA_LIBRARIES})

if (gw_profiling)
    find_library(NVTX_LIBRARY nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    target_compile_definitions(${MODULE_NAME} PUBLIC -DGW_PROFILING)
    target_link_libraries(${MODULE_NAME} PUBLIC ${NVTX_LIBRARY})
endif()

if (gw_device_synchronize_kernels)
    target_compile_definitions(${MODULE_NAME} PUBLIC GW_DEVICE_SYNCHRONIZE)
endif()

if(gw_enable_caching_allocator)
    target_compile_definitions(${MODULE_NAME} PUBLIC GW_ENABLE_CACHING_ALLOCATOR)
endif()

target_include_directories(${MODULE_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${CUDA_INCLUDE_DIRS}
)

install(TARGETS ${MODULE_NAME}
    EXPORT ${MODULE_NAME}
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${MODULE_NAME} DESTINATION cmake)

# Add documentation
add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add tests
add_subdirectory(tests)

# Adding formatting
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
