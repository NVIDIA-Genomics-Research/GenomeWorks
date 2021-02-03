

cmake_minimum_required(VERSION 3.10.2)

set(CUDAALIGNER_BENCHMARK_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")
configure_file(cudaaligner_file_location.hpp.in ${PROJECT_BINARY_DIR}/data/cudaaligner_file_location.hpp @ONLY)

set_property(GLOBAL PROPERTY cudaaligner_data_include_dir "${PROJECT_BINARY_DIR}/data")
