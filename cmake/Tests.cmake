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



#Cmake macro to initialzie ctest.
enable_testing()

get_property(enable_tests GLOBAL PROPERTY enable_tests)
function(gw_add_tests NAME SOURCES LIBS)
    # Add test executable
    if (enable_tests)
        CUDA_ADD_EXECUTABLE(${NAME} ${SOURCES})

        # Link gtest to tests binary
        target_link_libraries(${NAME}
            ${LIBS}
            gtest
            gmock)
        # Install to tests location
        install(TARGETS ${NAME}
            DESTINATION tests)
    endif()
endfunction(gw_add_tests)
