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



get_property(enable_benchmarks GLOBAL PROPERTY enable_benchmarks)
function(gw_add_benchmarks NAME MODULE SOURCES LIBS)
    # Add test executable
    if (enable_benchmarks)
        cuda_add_executable(${NAME} ${SOURCES})

        # Link gtest to benchmarks binary
        target_link_libraries(${NAME}
            ${LIBS}
            benchmark)
        # Install to benchmarks location
        install(TARGETS ${NAME}
            DESTINATION benchmarks/${MODULE})
    endif()
endfunction(gw_add_benchmarks)
