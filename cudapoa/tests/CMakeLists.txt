cmake_minimum_required(VERSION 3.10.2)
project(cudapoatests)

set(SOURCES
    main.cpp
    Test_CudapoaBatch.cpp
    Test_CudapoaTopSort.cpp)

set(LIBS
    cudapoa)

gw_add_tests(${PROJECT_NAME} "${SOURCES}" "${LIBS}")
