

get_property(enable_benchmarks GLOBAL PROPERTY enable_benchmarks)
function(gw_add_benchmarks NAME MODULE SOURCES LIBS)
    # Add test executable
    if (enable_benchmarks)
        cuda_add_executable(${NAME} ${SOURCES})

        # Link gtest to benchmarks binary
        target_link_libraries(${NAME}
            ${LIBS}
            benchmark)
        # Install to benchmarks location
        install(TARGETS ${NAME}
            DESTINATION benchmarks/${MODULE})
    endif()
endfunction(gw_add_benchmarks)
