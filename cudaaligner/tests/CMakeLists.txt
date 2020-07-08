

set(TARGET_NAME cudaalignertests)

set(SOURCES
    main.cpp
    cudaaligner_test_cases.cpp
    Test_Misc.cpp
    Test_AlignmentImpl.cpp
    Test_AlignerGlobal.cpp
    Test_MyersAlgorithm.cpp
    Test_HirschbergMyers.cu
    Test_NeedlemanWunschImplementation.cpp
    )

set(LIBS
    cudaaligner
    gwbase)

gw_add_tests(${TARGET_NAME} "${SOURCES}" "${LIBS}")
