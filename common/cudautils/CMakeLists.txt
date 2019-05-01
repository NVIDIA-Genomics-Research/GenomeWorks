project(cudautils)

add_library(cudautils INTERFACE)
target_link_libraries(cudautils INTERFACE logging)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/cudautils)

target_include_directories(cudautils
    INTERFACE 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS cudautils 
    EXPORT cudautils
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT cudautils DESTINATION cmake)
