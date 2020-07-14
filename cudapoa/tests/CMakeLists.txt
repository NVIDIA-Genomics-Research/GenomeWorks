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



set(TARGET_NAME cudapoatests)

set(SOURCES
    main.cpp
    Test_CudapoaBatch.cu
    Test_CudapoaTopSort.cu
    Test_CudapoaAddAlignment.cu
    Test_CudapoaNW.cu
    Test_CudapoaGenerateConsensus.cu
    Test_CudapoaBatchEnd2End.cu
    Test_CudapoaGenerateMSA2.cu
    Test_CudapoaSerializeGraph.cpp)

get_property(cudapoa_data_include_dir GLOBAL PROPERTY cudapoa_data_include_dir)
include_directories(${cudapoa_data_include_dir})

set(LIBS
    cudapoa
    spoa)

gw_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
