cmake_minimum_required(VERSION 3.10.2)
project(GenomeWorksCPP)

include(cmake/Doxygen.cmake)

set(CMAKE_CXX_STANDARD 17)

if (NOT TARGET bioparser)
    add_subdirectory(3rdparty/bioparser EXCLUDE_FROM_ALL)
endif()

add_subdirectory(cudapoa)
add_subdirectory(cudamapper)

add_docs_target("GenomeWorks" "1.0")
