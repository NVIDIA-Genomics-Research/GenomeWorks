

set(CUDAPOA_BENCHMARK_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../data")
configure_file(file_location.hpp.in ${PROJECT_BINARY_DIR}/data/file_location.hpp @ONLY)

set_property(GLOBAL PROPERTY cudapoa_data_include_dir "${PROJECT_BINARY_DIR}/data")
