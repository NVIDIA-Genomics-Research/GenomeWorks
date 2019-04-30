project(cudamapper)

add_library(indexer
        src/index_generator_cpu.cpp)
target_include_directories(indexer PUBLIC include)
target_link_libraries(indexer bioparser)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/cudamapper)

add_executable(cudamapper
        src/cudamapper.cpp
        src/main.cpp
        src/sequence.cpp
        src/bioparser_sequence.cpp
        src/index.cpp
        src/index_gpu.cpp
        src/index_generator.cpp
        src/index_generator_cpu.cpp
        src/minimizer.cpp
        )

target_include_directories(cudamapper
    PUBLIC 
        $<INSTALL_INTERFACE:include>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(cudamapper bioparser indexer)

# Add tests folder
add_subdirectory(tests)

install(TARGETS cudamapper 
    EXPORT cudamapper
    DESTINATION lib
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT cudamapper DESTINATION cmake)
