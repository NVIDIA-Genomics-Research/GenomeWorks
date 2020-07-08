

project(gwio)

get_property(gw_library_type GLOBAL PROPERTY gw_library_type)
add_library(${PROJECT_NAME} ${gw_library_type}
        src/fasta_parser.cpp
        src/kseqpp_fasta_parser.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC gwbase z)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/claraparabricks/genomeworks/io)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${KSEQPP_DIR}
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_options(${PROJECT_NAME} PRIVATE -Werror)

install(TARGETS ${PROJECT_NAME} 
    EXPORT ${PROJECT_NAME}
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${PROJECT_NAME} DESTINATION cmake)

# Add auto formatting.
gw_enable_auto_formatting("${CMAKE_CURRENT_SOURCE_DIR}")
