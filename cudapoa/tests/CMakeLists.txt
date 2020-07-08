

set(TARGET_NAME cudapoatests)

set(SOURCES
    main.cpp
    Test_CudapoaBatch.cu
    Test_CudapoaTopSort.cu
    Test_CudapoaAddAlignment.cu
    Test_CudapoaNW.cu
    Test_CudapoaGenerateConsensus.cu
    Test_CudapoaBatchEnd2End.cu
    Test_CudapoaGenerateMSA2.cu
    Test_CudapoaSerializeGraph.cpp)

get_property(cudapoa_data_include_dir GLOBAL PROPERTY cudapoa_data_include_dir)
include_directories(${cudapoa_data_include_dir})

set(LIBS
    cudapoa
    spoa)

gw_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
