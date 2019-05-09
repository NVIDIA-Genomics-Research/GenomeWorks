cmake_minimum_required(VERSION 3.10.2)
project(GenomeWorksCPP)

# Process options.
option(gw_enable_tests "Build GenomeWorks unit tests" OFF)

if (gw_enable_tests)
    message(STATUS "Enabling GenomeWorks unit tests")
    set_property(GLOBAL PROPERTY enable_tests ON)
endif()

include(cmake/Doxygen.cmake)
include(cmake/3rdparty.cmake)
include(cmake/Tests.cmake)

set(CMAKE_CXX_STANDARD 17)

# Add GenomeWorks projects.
add_subdirectory(common/cudautils)
add_subdirectory(common/logging)
add_subdirectory(common/utils)
add_subdirectory(cudapoa)
add_subdirectory(cudamapper)
add_subdirectory(cudaaligner)

set_doxygen_mainpage(${CMAKE_CURRENT_SOURCE_DIR}/docs/README-SDK.md)
add_docs_target("GenomeWorks" "1.0")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX ../deploy CACHE PATH "The default SDK install path is ../deploy" FORCE)
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
