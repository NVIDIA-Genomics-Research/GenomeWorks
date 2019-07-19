# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Declare core CUDA structs and API calls which 
# can be used to interface with the CUDA runtime API.

# Declare cudaStream_t and cudaError_t without specifying particular
# include files since they will be added by the CUDA runtime API includes.
cdef extern from *:
    ctypedef void* _Stream "cudaStream_t"
    ctypedef int _Error "cudaError_t"

# Declare commonly used CUDA runtime API calls.
cdef extern from "cuda_runtime_api.h":
    # CUDA Stream APIs
    cdef _Error cudaStreamCreate(_Stream* s)
    cdef _Error cudaStreamDestroy(_Stream s)
    cdef _Error cudaStreamSynchronize(_Stream s)
    # CUDA Error APIs
    cdef _Error cudaGetLastError()
    cdef const char* cudaGetErrorString(_Error e)
    cdef const char* cudaGetErrorName(_Error e)
