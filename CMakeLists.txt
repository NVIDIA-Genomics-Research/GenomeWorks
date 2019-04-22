cmake_minimum_required(VERSION 3.10.2)
project(GenomeWorksCPP)

include(cmake/Doxygen.cmake)

set(CMAKE_CXX_STANDARD 17)

if (NOT TARGET bioparser)
    add_subdirectory(3rdparty/bioparser EXCLUDE_FROM_ALL)
endif()

add_subdirectory(cudapoa)
add_subdirectory(cudamapper)

set_doxygen_mainpage(${CMAKE_CURRENT_SOURCE_DIR}/docs/README-SDK.md)
add_docs_target("GenomeWorks" "1.0")

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX ../deploy CACHE PATH "The default SDK install path is ../deploy" FORCE)
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
