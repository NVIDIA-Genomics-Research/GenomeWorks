cmake_minimum_required(VERSION 3.10.2)
project(cudamappertests)

set(SOURCES
    main.cpp)

set(LIBS
    indexer)

gw_add_tests(${PROJECT_NAME} ${SOURCES} ${LIBS})
