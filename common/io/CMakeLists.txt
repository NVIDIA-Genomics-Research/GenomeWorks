#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

project(gwio)

get_property(gw_library_type GLOBAL PROPERTY gw_library_type)
add_library(${PROJECT_NAME} ${gw_library_type}
        src/fasta_parser.cpp
        src/kseqpp_fasta_parser.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC gwbase z)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/claragenomics/io)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${KSEQPP_DIR}
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(${PROJECT_NAME} PRIVATE -Werror)

install(TARGETS ${PROJECT_NAME} 
    EXPORT ${PROJECT_NAME}
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${PROJECT_NAME} DESTINATION cmake)

# Add auto formatting.
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
