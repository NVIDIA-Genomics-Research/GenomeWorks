project(cudamapper)

set(CMAKE_CXX_STANDARD 17)

add_executable(cudamapper
        src/main.cpp
        src/sequence.cpp
        src/index.cpp)

target_include_directories(cudamapper PUBLIC include)
target_link_libraries(cudamapper bioparser)
