# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport int8_t, int16_t, uint16_t, int32_t
from claragenomics.bindings.cuda cimport _Stream, _Error

cdef extern from "cudapoa/cudapoa.hpp" namespace "claragenomics::cudapoa":
    cdef enum StatusType:
        success = 0
        exceeded_maximum_poa
        exceeded_maximum_sequence_size
        exceeded_maximum_sequences_per_poa
        node_count_exceeded_maximum_graph_size
        seq_len_exceeded_maximum_nodes_per_window
        loop_count_exceeded_upper_bound
        generic_error

    cdef enum OutputType:
        consensus = 0x1
        msa = 0x1 << 1

    cdef StatusType Init()

cdef extern from "cudapoa/batch.hpp" namespace "claragenomics::cudapoa":
    cdef cppclass Batch:
        StatusType add_poa() except +
        StatusType add_seq_to_poa(char*, int8_t*, int32_t) except +
        void generate_poa() except +
        void get_msa(vector[vector[string]]&, vector[StatusType]&) except +
        void get_consensus(vector[string]&, vector[vector[uint16_t]]&, vector[StatusType]&) except +
        int get_total_poas() except +
        void set_cuda_stream(_Stream) except +

    cdef unique_ptr[Batch] create_batch(int32_t, int32_t, int32_t, int8_t, int16_t, int16_t, int16_t, bool)
