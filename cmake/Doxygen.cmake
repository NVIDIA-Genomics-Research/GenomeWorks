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



set_property(GLOBAL PROPERTY doxygen_src_dirs)
set_property(GLOBAL PROPERTY doxygen_mainpage)
function(add_doxygen_source_dir)
    get_property(tmp GLOBAL PROPERTY doxygen_src_dirs)
    # avoid leading space in path
    foreach(arg ${ARGV})
        if("${tmp}" STREQUAL "")
            set(tmp "${arg}")
        else()
            set(tmp "${tmp}\;${arg}")
        endif()
    endforeach()

    set_property(GLOBAL PROPERTY doxygen_src_dirs "${tmp}")
endfunction(add_doxygen_source_dir)

function(set_doxygen_mainpage page)
    set_property(GLOBAL PROPERTY doxygen_mainpage ${page})
    add_doxygen_source_dir(${page})
endfunction(set_doxygen_mainpage)

function(add_docs_target DOXY_PACKAGE_NAME DOXY_PACKAGE_VERSION)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_PROJECT_NAME ${DOXY_PACKAGE_NAME})
        set(DOXYGEN_PROJECT_NUMBER ${DOXY_PACKAGE_VERSION})
        set(DOXYGEN_WARN_AS_ERROR TRUE)

        get_property(final_doxygen_mainpage GLOBAL PROPERTY doxygen_mainpage)

        set(DOXYGEN_USE_MDFILE_AS_MAINPAGE ${final_doxygen_mainpage})

        get_property(final_doxygen_src_dirs GLOBAL PROPERTY doxygen_src_dirs)
        doxygen_add_docs(docs ALL ${final_doxygen_src_dirs})
        set_target_properties(docs PROPERTIES EXCLUDE_FROM_ALL NO)
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/ DESTINATION docs)
    else()
        message(STATUS "Doxygen not found. Doc gen disabled.")
    endif()

endfunction(add_docs_target)
