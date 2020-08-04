

cmake_minimum_required(VERSION 3.10.2)

set(CUDAMAPPER_BENCHMARK_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")
configure_file(cudamapper_file_location.hpp.in ${PROJECT_BINARY_DIR}/data/cudamapper_file_location.hpp @ONLY)

set_property(GLOBAL PROPERTY cudamapper_data_include_dir "${PROJECT_BINARY_DIR}/data")
