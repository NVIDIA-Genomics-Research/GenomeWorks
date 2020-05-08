#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport int8_t, int16_t, uint16_t, int32_t
from libcpp.vector cimport vector

# NOTE: The mixup between Python bool type and C++ bool type leads to very nasty
# bugs in cython. If the Python bool type is used in pxd file and a bool value is
# supplied as arg to bingins, it's always interpreted as True. The libcpp bool object
# handles bool values passthrough from Python -> C correctly, and therefore must always
# be used.
from libcpp cimport bool as c_bool

from bindings.cuda_runtime_api cimport _Stream
from bindings.graph cimport DirectedGraph

# This file declares public structs and API calls
# from the ClaraGenomicsAnalysis `cudapoa` module.

# Declare structs and APIs from cudapoa.hpp.
cdef extern from "claragenomics/cudapoa/cudapoa.hpp" namespace "claragenomics::cudapoa":
    cdef enum StatusType:
        success = 0
        exceeded_maximum_poas
        exceeded_maximum_sequence_size
        exceeded_maximum_sequences_per_poa
        node_count_exceeded_maximum_graph_size
        edge_count_exceeded_maximum_graph_size
        seq_len_exceeded_maximum_nodes_per_window
        loop_count_exceeded_upper_bound
        output_type_unavailable
        generic_error

    cdef enum OutputType:
        consensus = 0x1
        msa = 0x1 << 1

    cdef StatusType Init()

# Declare structs and APIs from batch.hpp.
cdef extern from "claragenomics/cudapoa/batch.hpp" namespace "claragenomics::cudapoa":
    cdef struct Entry:
        const char* seq
        const int8_t* weights
        int32_t length

    cdef cppclass BatchSize:
        int32_t max_sequence_size
        int32_t max_concensus_size
        int32_t max_nodes_per_window
        int32_t max_nodes_per_window_banded
        int32_t max_matrix_graph_dimension
        int32_t max_matrix_graph_dimension_banded
        int32_t max_matrix_sequence_dimension
        int32_t max_sequences_per_poa

        BatchSize(int32_t, int32_t)
        BatchSize(int32_t, int32_t,
                  int32_t, int32_t, int32_t)

    ctypedef vector[Entry] Group

    cdef cppclass Batch:
        StatusType add_poa_group(vector[StatusType]&, const Group&) except +
        void generate_poa() except +
        StatusType get_msa(vector[vector[string]]&, vector[StatusType]&) except +
        StatusType get_consensus(vector[string]&, vector[vector[uint16_t]]&, vector[StatusType]&) except +
        void get_graphs(vector[DirectedGraph]&, vector[StatusType]&) except +
        int get_total_poas() except +
        int batch_id() except +
        void reset() except +

    cdef unique_ptr[Batch] create_batch(int32_t, _Stream, size_t, int8_t,
                                        const BatchSize&, int16_t, int16_t,
                                        int16_t, c_bool)
