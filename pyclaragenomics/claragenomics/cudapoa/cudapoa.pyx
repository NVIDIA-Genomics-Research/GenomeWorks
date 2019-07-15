# distutils: language = c++

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport int8_t, int16_t, uint16_t, int32_t
from cudapoa cimport StatusType, OutputType, Batch, create_batch

cdef class PyCudapoa:
    cdef unique_ptr[Batch] my_test

    def __cinit__(self):
        self.my_test = create_batch(100, 100, 0, 0x1, -8, -6, 8, False)

    def add_poa(self):
        return deref(self.my_test).add_poa()

    def get_total_poas(self):
        return deref(self.my_test).get_total_poas()

    def add_seq_to_poa(self, seq):
        cdef bytes py_bytes = seq.encode()
        cdef char* c_string = py_bytes
        return deref(self.my_test).add_seq_to_poa(c_string, NULL, len(seq))

    def generate_poa(self):
        deref(self.my_test).generate_poa()

    def get_msa(self):
        cdef vector[vector[string]] msa
        cdef vector[StatusType] status
        deref(self.my_test).get_msa(msa, status)
        return (msa, status)

    def get_consensus(self):
        cdef vector[string] con
        cdef vector[vector[uint16_t]] cov
        cdef vector[StatusType] s
        deref(self.my_test).get_consensus(con, cov, s)
        return con
