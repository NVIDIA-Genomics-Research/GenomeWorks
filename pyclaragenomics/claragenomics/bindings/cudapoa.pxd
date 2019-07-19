# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport int8_t, int16_t, uint16_t, int32_t
from claragenomics.bindings.cuda cimport _Stream

# This file declares public structs and API calls 
# from the ClaraGenomicsAnalysis `cudapoa` module.

# Declare structs and APIs from cudapoa.hpp.
cdef extern from "cudapoa/cudapoa.hpp" namespace "claragenomics::cudapoa":
    cdef enum StatusType:
        success = 0
        exceeded_maximum_poa
        exceeded_maximum_sequence_size
        exceeded_maximum_sequences_per_poa
        node_count_exceeded_maximum_graph_size
        seq_len_exceeded_maximum_nodes_per_window
        loop_count_exceeded_upper_bound
        output_type_unavailable
        generic_error

    cdef enum OutputType:
        consensus = 0x1
        msa = 0x1 << 1

    cdef StatusType Init()

# Declare structs and APIs from batch.hpp.
cdef extern from "cudapoa/batch.hpp" namespace "claragenomics::cudapoa":
    cdef cppclass Batch:
        StatusType add_poa() except +
        StatusType add_seq_to_poa(char*, int8_t*, int32_t) except +
        void generate_poa() except +
        StatusType get_msa(vector[vector[string]]&, vector[StatusType]&) except +
        StatusType get_consensus(vector[string]&, vector[vector[uint16_t]]&, vector[StatusType]&) except +
        int get_total_poas() except +

    cdef unique_ptr[Batch] create_batch(int32_t, int32_t, _Stream, int32_t, int8_t, int16_t, int16_t, int16_t, bool)
