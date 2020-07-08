

# Check CUDA dependency for project.
find_package(CUDA 9.0 REQUIRED)

if(NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
else()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -use_fast_math -Xcompiler -Wall,-Wno-pedantic")

    if (${CUDA_VERSION_MAJOR} VERSION_LESS "10")
        set(gw_cuda_before_10_0 TRUE)
    else()
        set(gw_cuda_before_10_0 FALSE)
    endif()

    if (${CUDA_VERSION_STRING} VERSION_GREATER "10")
        set(gw_cuda_after_10_0 TRUE)
    else()
        set(gw_cuda_after_10_0 FALSE)
    endif()

endif()

