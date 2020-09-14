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



# Formatting build targets

# list of symbols to fill with commands to run auto-formatter
define_property(GLOBAL PROPERTY GW_FORMAT_SOURCES
    BRIEF_DOCS "List of files for autoformatting"
    FULL_DOCS "List of files for autoformatting"
)

#
# add a specific project to a list of auto-formated code
#
# param(DIRECTORY) - directory to activate formatting for
#
function(gw_enable_auto_formatting DIRECTORY)
    if(NOT (IS_ABSOLUTE ${DIRECTORY} AND IS_DIRECTORY ${DIRECTORY}))
        message(FATAL_ERROR "Formatting: '${DIRECTORY}' not an absolute path to a directory")
    endif()

    file(GLOB_RECURSE FOLDER_FORMAT_SOURCES "${DIRECTORY}/*.h"
                                            "${DIRECTORY}/*.hpp"
                                            "${DIRECTORY}/*.cpp"
                                            "${DIRECTORY}/*.cc"
                                            "${DIRECTORY}/*.c"
                                            "${DIRECTORY}/*.cuh"
                                            "${DIRECTORY}/*.cu")

    set_property(GLOBAL APPEND PROPERTY GW_FORMAT_SOURCES "${FOLDER_FORMAT_SOURCES}")
endfunction()

# activate formatting
function(gw_enable_formatting_targets)
    find_program(GW_CLANGFORMAT_EXECUTABLE NAMES clang-format)
    if(GW_CLANGFORMAT_EXECUTABLE)

        get_property(FOLDER_FORMAT_SOURCES GLOBAL PROPERTY GW_FORMAT_SOURCES)

        set(format_list "")
        set(format_check_list "")

        foreach(source IN LISTS FOLDER_FORMAT_SOURCES)
            file(RELATIVE_PATH RELATIVE_PATH ${PROJECT_SOURCE_DIR} ${source})

            # Apply formatting
            get_filename_component(SYMBOL_BASE_DIR ${RELATIVE_PATH} DIRECTORY)
            file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/formatting/${SYMBOL_BASE_DIR})
            add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/formatting/${RELATIVE_PATH}-formatted
                COMMAND "${GW_CLANGFORMAT_EXECUTABLE}" "-i" "-style=file" ${source}
                COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/formatting/${RELATIVE_PATH}-formatted
                DEPENDS ${source}
                COMMENT "Formatting ${source}")

            list(APPEND format_list ${CMAKE_BINARY_DIR}/formatting/${RELATIVE_PATH}-formatted)

            # Check / show format differences (check will terminate on error)
            add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/formatting/${RELATIVE_PATH}-format-checked
                COMMAND ${GW_CLANGFORMAT_EXECUTABLE} ${source} | diff -u ${source} -
                && touch ${CMAKE_BINARY_DIR}/formatting/${RELATIVE_PATH}-format-checked
                DEPENDS ${source}
                COMMENT "Checking format of ${source}")
            add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/formatting/${RELATIVE_PATH}-format-shown
                COMMAND if ${GW_CLANGFORMAT_EXECUTABLE} ${source} | diff -u ${source} - \; then
                touch ${CMAKE_BINARY_DIR}/formatting/${RELATIVE_PATH}-format-checked\;
                fi
                DEPENDS ${source}
                COMMENT "Showing format of ${source}")

            list(APPEND format_check_list ${CMAKE_BINARY_DIR}/formatting/${RELATIVE_PATH}-format-checked)
        endforeach()

        add_custom_target(format DEPENDS ${format_list} COMMENT "Format source files")
        add_custom_target(check-format DEPENDS ${format_check_list} COMMENT "Check format of source files")
    else()
        message(STATUS "clang-format not found. Auto-formatting disabled.")
    endif()

endfunction()
