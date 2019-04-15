project(cudamapper)

add_library(indexer
        src/cpuindex.cpp)
target_include_directories(indexer PUBLIC include)
target_link_libraries(indexer bioparser)

add_executable(cudamapper
        src/main.cpp
        src/sequence.cpp)
target_include_directories(cudamapper PUBLIC include)
target_link_libraries(cudamapper bioparser indexer)
