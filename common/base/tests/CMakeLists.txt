

set(TARGET_NAME gwbasetests)

set(SOURCES
    main.cpp
    Test_UtilsCudasort.cu
    Test_UtilsThreadsafeContainers.cpp
    TestGraph.cpp
    Test_GenomeUtils.cpp)

set(LIBS
    gwbase
    cub)

gw_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
