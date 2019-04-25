cmake_minimum_required(VERSION 3.10.2)

#Cmake macro to initialzie ctest.
enable_testing()

function(gw_add_tests NAME SOURCES LIBS)
    # Add test executable
    add_executable(${NAME} ${SOURCES})
    # Link gtest to tests binary
    target_link_libraries(${NAME}
        ${LIBS}
        gtest)
    # Install to tests location
    install(TARGETS ${NAME}
        DESTINATION tests)
endfunction(gw_add_tests)
