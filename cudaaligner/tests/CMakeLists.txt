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



set(TARGET_NAME cudaalignertests)

set(SOURCES
    main.cpp
    cudaaligner_test_cases.cpp
    Test_Misc.cpp
    Test_AlignmentImpl.cpp
    Test_AlignerGlobal.cpp
    Test_ApproximateBandedMyers.cpp
    Test_MyersAlgorithm.cu
    Test_HirschbergMyers.cu
    Test_NeedlemanWunschImplementation.cpp
    )

set(LIBS
    cudaaligner
    gwbase)

gw_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
