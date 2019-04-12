cmake_minimum_required(VERSION 3.10.2)
project(GenomeWorksCPP)

if (NOT TARGET bioparser)
    add_subdirectory(3rdparty/bioparser EXCLUDE_FROM_ALL)
endif()

add_subdirectory(cudapoa)
add_subdirectory(cudamapper)

