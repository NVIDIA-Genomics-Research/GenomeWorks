from claragenomics.bindings.cuda cimport *

cdef class CudaStream:
    cdef size_t stream

    def __cinit__(self):
        cdef _Stream s
        cdef _Error e = cudaStreamCreate(&s)
        if (e != 0):
            raise RuntimeError("Can't construct")
        self.stream = <size_t>s

    def __dealloc__(self):
        self.sync()
        cdef _Stream s = <_Stream>self.stream
        cdef _Error e = cudaStreamDestroy(s)
        if (e != 0):
            raise RuntimeError("Can't destroy")

    def sync(self):
        cdef _Stream s = <_Stream>self.stream
        cdef _Error e = cudaStreamSynchronize(s)
        if (e != 0):
            raise RuntimeError("Can't sync")

    def get_stream(self):
        return self.stream
