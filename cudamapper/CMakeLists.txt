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



set(MODULE_NAME cudamapper)

# Process data subdirectory first
add_subdirectory(data)

GitVersion()

configure_file(${PROJECT_SOURCE_DIR}/common/base/src/version.cpp.in
                ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

find_package(CUDA 9.0 QUIET REQUIRED)

if(NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
else()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo --expt-extended-lambda -use_fast_math -Xcompiler -Wall,-Wno-pedantic -std=c++14")
    # Determine arch flags
    validate_boolean(gw_cuda_gen_all_arch)
    if (gw_cuda_gen_all_arch)
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Common")
    else()
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
    endif()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")
    message(STATUS "nvcc flags for ${MODULE_NAME}: ${CUDA_NVCC_FLAGS}")
endif()

cuda_add_library(${MODULE_NAME}
        src/application_parameters.cpp
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
        src/overlapper_triggered.cu
        src/overlapper_anchmer.cu
        src/overlapper_minimap.cu
        src/utils.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/version.cpp)

target_include_directories(${MODULE_NAME}
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(${MODULE_NAME} gwbase gwio cub)
target_compile_options(${MODULE_NAME} PRIVATE -Werror)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

cuda_add_executable(${MODULE_NAME}-bin
        src/main.cu
)

target_compile_options(${MODULE_NAME}-bin PRIVATE -Werror)
target_link_libraries(${MODULE_NAME}-bin ${MODULE_NAME} cudaaligner)
set_target_properties(${MODULE_NAME}-bin PROPERTIES OUTPUT_NAME ${MODULE_NAME})


# Add tests folder
add_subdirectory(tests)
add_subdirectory(samples)

install(TARGETS ${MODULE_NAME}
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

# Add auto formatting.
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
