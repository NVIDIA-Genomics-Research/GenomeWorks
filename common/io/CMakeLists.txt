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



set(MODULE_NAME gwio)

get_property(gw_library_type GLOBAL PROPERTY gw_library_type)
add_library(${MODULE_NAME} ${gw_library_type}
        src/fasta_parser.cpp
        src/kseqpp_fasta_parser.cpp)
target_link_libraries(${MODULE_NAME} PUBLIC gwbase z)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/claraparabricks/genomeworks/io)

target_include_directories(${MODULE_NAME}
    PRIVATE
        ${KSEQPP_DIR}
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(${MODULE_NAME} PRIVATE -Werror)

install(TARGETS ${MODULE_NAME} 
    EXPORT ${MODULE_NAME}
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${MODULE_NAME} DESTINATION cmake)

# Add auto formatting.
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
