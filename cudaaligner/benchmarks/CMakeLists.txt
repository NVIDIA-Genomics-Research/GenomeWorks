


project(benchmark_cudaaligner)

set(SOURCES
    main.cpp
    )

set(LIBS
    cudaaligner_internal
    cudaaligner
    gwbase)

gw_add_benchmarks(${PROJECT_NAME} "cudaaligner" "${SOURCES}" "${LIBS}")

install(FILES README.md
    DESTINATION benchmarks/cudaaligner)
