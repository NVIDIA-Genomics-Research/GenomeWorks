#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

project(utils)


get_property(cga_library_type GLOBAL PROPERTY cga_library_type)
add_library(utils ${cga_library_type}
        src/cudautils.cpp)
target_link_libraries(utils PUBLIC cgaconfig logging)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(utils
    PUBLIC
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS utils
    EXPORT utils
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT utils DESTINATION cmake)

# Add tests
add_subdirectory(tests)

# Adding formatting
cga_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
