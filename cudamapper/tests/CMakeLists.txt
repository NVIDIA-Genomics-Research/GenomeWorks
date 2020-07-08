

set(TARGET_NAME cudamappertests)

set(SOURCES
    main.cpp
    Test_CudamapperIndexBatcher.cu
    Test_CudamapperIndexCache.cu
    Test_CudamapperIndexDescriptor.cpp
    Test_CudamapperIndexGPU.cu
    Test_CudamapperMatcherGPU.cu
    Test_CudamapperMinimizer.cpp
    Test_CudamapperOverlapper.cpp
    Test_CudamapperOverlapperTriggered.cu
    Test_CudamapperUtilsKmerFunctions.cpp
   )

get_property(cudamapper_data_include_dir GLOBAL PROPERTY cudamapper_data_include_dir)
include_directories(${cudamapper_data_include_dir})

set(LIBS cudamapper)

gw_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
