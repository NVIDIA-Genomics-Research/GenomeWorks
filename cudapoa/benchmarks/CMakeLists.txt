

## Add benchmark to cudapoa project

project(benchmark_cudapoa)

set(SOURCES
        main.cpp
        )
get_property(cudapoa_data_include_dir GLOBAL PROPERTY cudapoa_data_include_dir)
include_directories(${cudapoa_data_include_dir})

set(LIBS
        cudapoa)

gw_add_benchmarks(${PROJECT_NAME} "cudapoa" "${SOURCES}" "${LIBS}")

install(FILES README.md
    DESTINATION benchmarks/cudapoa)
