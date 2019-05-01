project(logging)

add_library(logging
        src/logging.cpp)
target_link_libraries(logging PUBLIC spdlog pthread)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/logging)

target_include_directories(logging
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS logging 
    EXPORT logging
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT logging DESTINATION cmake)
