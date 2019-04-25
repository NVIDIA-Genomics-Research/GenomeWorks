cmake_minimum_required(VERSION 3.10.2)

#Cmake macro to initialzie ctest.
enable_testing()

get_property(enable_tests GLOBAL PROPERTY enable_tests)
function(gw_add_tests NAME SOURCES LIBS)
    # Add test executable
    if (enable_tests)
        add_executable(${NAME} ${SOURCES})
        # Link gtest to tests binary
        target_link_libraries(${NAME}
            ${LIBS}
            gtest)
        # Install to tests location
        install(TARGETS ${NAME}
            DESTINATION tests)
    endif()
endfunction(gw_add_tests)
