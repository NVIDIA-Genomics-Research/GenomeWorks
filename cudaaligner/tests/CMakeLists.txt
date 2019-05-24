cmake_minimum_required(VERSION 3.10.2)
project(cudaalignertests)

set(SOURCES
    main.cpp
    Test_AlignmentImpl.cpp
    Test_AlignerGlobal.cpp
    Test_NeedlemanWunschImplementation.cpp)

set(LIBS
    cudaaligner)

gw_add_tests(${PROJECT_NAME} "${SOURCES}" "${LIBS}")
