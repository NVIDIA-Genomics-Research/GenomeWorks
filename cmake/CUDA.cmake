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



# Check CUDA dependency for project.
find_package(CUDA 9.0 REQUIRED)

if(NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
else()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -use_fast_math -Xcompiler -Wall,-Wno-pedantic")

    if (${CUDA_VERSION_MAJOR} VERSION_LESS "10")
        set(gw_cuda_before_10_0 TRUE)
    else()
        set(gw_cuda_before_10_0 FALSE)
    endif()

    if (${CUDA_VERSION_STRING} VERSION_GREATER "10")
        set(gw_cuda_after_10_0 TRUE)
    else()
        set(gw_cuda_after_10_0 FALSE)
    endif()

endif()

