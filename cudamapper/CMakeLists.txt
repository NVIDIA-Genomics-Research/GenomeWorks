project(cudamapper)

add_library(indexer
        src/cpu_index.cpp)
target_include_directories(indexer PUBLIC include)
target_link_libraries(indexer bioparser)

add_doxygen_source_dir(${CMAKE_CURRENT_SOURCE_DIR}/include/cudamapper)

add_executable(cudamapper
        src/main.cpp
        src/sequence.cpp
        src/bioparser_sequence.cpp
        src/index.cpp
        src/cpu_index.cpp
        )
target_include_directories(cudamapper PUBLIC include)
target_link_libraries(cudamapper bioparser indexer)
