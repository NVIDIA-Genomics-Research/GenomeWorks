project(cudamapper)

add_executable(cudamapper
        src/main.cpp
        src/sequence.cpp
        src/index.cpp)

set_property(TARGET cudamapper PROPERTY CMAKE_CXX_STANDARD 17)

target_include_directories(cudamapper PUBLIC include)
target_link_libraries(cudamapper bioparser)
