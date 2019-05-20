cmake_minimum_required(VERSION 3.10.2)
project(cudamappertests)

set(SOURCES
    main.cpp
    Test_CudamapperIndexGPU.cpp)

set(LIBS
    index_generator_cpu
    index_gpu)

gw_add_tests(${PROJECT_NAME} "${SOURCES}" "${LIBS}")
