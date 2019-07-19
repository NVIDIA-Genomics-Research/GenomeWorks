from claragenomics.bindings.cuda cimport *

cdef class CudaStream:
    """
    Class to abstract the usage of CUDA streams which enable
    easy asynchronous execution of CUDA kernels.
    """

    # Using size_t to store stream since underlying
    # representation of cudaStream_t is a (void *)
    # and python doesn't know how to deal with converting
    # (void *) to python objects.
    cdef size_t stream

    def __cinit__(self):
        cdef _Stream s
        cdef _Error e = cudaStreamCreate(&s)
        cdef bytes error_str
        if (e != 0):
            error_str = cudaGetErrorString(e)
            raise RuntimeError(b'Cannot create stream: ' + error_str)
        self.stream = <size_t>s

    def __dealloc__(self):
        self.sync()
        cdef _Stream s = <_Stream>self.stream
        cdef _Error e = cudaStreamDestroy(s)
        cdef bytes error_str
        if (e != 0):
            error_str = cudaGetErrorString(e)
            raise RuntimeError(b'Cannot destroy stream:' + error_str)

    def sync(self):
        """
        Synchronize the CUDA stream.
        """
        cdef _Stream s = <_Stream>self.stream
        cdef _Error e = cudaStreamSynchronize(s)
        cdef bytes error_str
        if (e != 0):
            error_str = cudaGetErrorString(e)
            raise RuntimeError(b'Cannot sync stream: ' + error_str)

    def get_stream(self):
        return self.stream
