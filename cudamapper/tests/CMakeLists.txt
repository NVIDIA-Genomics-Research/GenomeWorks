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



set(TARGET_NAME cudamappertests)

set(SOURCES
    main.cpp
    Test_CudamapperIndexBatcher.cu
    Test_CudamapperIndexCache.cu
    Test_CudamapperIndexDescriptor.cpp
    Test_CudamapperIndexGPU.cu
    Test_CudamapperMatcherGPU.cu
    Test_CudamapperMinimizer.cpp
    Test_CudamapperOverlapper.cpp
    Test_CudamapperOverlapperTriggered.cu
    Test_CudamapperOverlapperMinimap.cu
    Test_CudamapperUtilsKmerFunctions.cpp
   )


get_property(cudamapper_data_include_dir GLOBAL PROPERTY cudamapper_data_include_dir)
include_directories(${cudamapper_data_include_dir})

set(LIBS cudamapper)

gw_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
