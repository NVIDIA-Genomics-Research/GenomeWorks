cmake_minimum_required(VERSION 3.10.2)
project(cudaalignertests)

set(SOURCES
    main.cpp)

set(LIBS
    cudaaligner)

gw_add_tests(${PROJECT_NAME} ${SOURCES} ${LIBS})
