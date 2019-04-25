cmake_minimum_required(VERSION 3.10.2)

# Add 3rd party build dependencies.
if (NOT TARGET bioparser)
    add_subdirectory(3rdparty/bioparser EXCLUDE_FROM_ALL)
endif()

get_property(enable_tests GLOBAL PROPERTY enable_tests)
if (enable_tests AND NOT TARGET gtest)
    add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
endif()

