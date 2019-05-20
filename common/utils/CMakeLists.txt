project(utils)

add_library(utils INTERFACE)
target_link_libraries(utils INTERFACE logging)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/utils)

target_include_directories(utils
    INTERFACE 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS utils 
    EXPORT utils
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT utils DESTINATION cmake)

# Adding formatting
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
