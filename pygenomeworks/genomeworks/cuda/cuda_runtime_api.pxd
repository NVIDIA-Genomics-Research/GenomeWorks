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


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Declare core CUDA structs and API calls which
# can be used to interface with the CUDA runtime API.

# Declare cudaStream_t and cudaError_t without specifying particular
# include files since they will be added by the CUDA runtime API includes.
cdef extern from *:
    ctypedef void* _Stream "cudaStream_t"
    ctypedef int _Error "cudaError_t"

# Declare commonly used CUDA runtime API calls.
cdef extern from "cuda_runtime_api.h":
    # CUDA Stream APIs
    cdef _Error cudaStreamCreate(_Stream* s)
    cdef _Error cudaStreamDestroy(_Stream s)
    cdef _Error cudaStreamSynchronize(_Stream s)
    # CUDA Error APIs
    cdef _Error cudaGetLastError()
    cdef const char* cudaGetErrorString(_Error e)
    cdef const char* cudaGetErrorName(_Error e)
    # CUDA Device Info APIs
    cdef _Error cudaGetDeviceCount(int* count)
    cdef _Error cudaSetDevice(int device)
    cdef _Error cudaGetDevice(int* device)
    cdef _Error cudaMemGetInfo(size_t* free, size_t* total)
