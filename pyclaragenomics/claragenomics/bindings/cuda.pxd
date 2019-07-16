# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

cdef extern from *:
    ctypedef void* _Stream "cudaStream_t"
    ctypedef int _Error "cudaError_t"

cdef extern from "cuda_runtime_api.h":
    cdef _Error cudaStreamCreate(_Stream* s)
    cdef _Error cudaStreamDestroy(_Stream s)
    cdef _Error cudaStreamSynchronize(_Stream s)
    cdef _Error cudaGetLastError()
    cdef const char* cudaGetErrorString(_Error e)
    cdef const char* cudaGetErrorName(_Error e)
