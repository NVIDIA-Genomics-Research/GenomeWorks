#
# Copyright 2019-2020 NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from genomeworks.cuda.cuda_runtime_api cimport _Stream
from genomeworks.cudapoa.graph cimport DirectedGraph

# This file declares public structs and API calls
# from the GenomeWorks `cudapoa` module.

# Declare structs and APIs from cudapoa.hpp.
cdef extern from "claraparabricks/genomeworks/cudapoa/cudapoa.hpp" namespace "claraparabricks::genomeworks::cudapoa":
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

    cdef enum BandMode:
        full_band = 0
        static_band
        adaptive_band

    cdef enum OutputType:
        consensus = 0x1
        msa = 0x1 << 1

    cdef StatusType Init()

# Declare structs and APIs from batch.hpp.
cdef extern from "claraparabricks/genomeworks/cudapoa/batch.hpp" namespace "claraparabricks::genomeworks::cudapoa":
    cdef struct Entry:
        const char* seq
        const int8_t* weights
        int32_t length

    cdef cppclass BatchConfig:
        int32_t max_sequence_size
        int32_t max_consensus_size
        int32_t max_nodes_per_graph
        int32_t max_nodes_per_graph_banded
        int32_t max_matrix_graph_dimension
        int32_t max_matrix_graph_dimension_banded
        int32_t max_matrix_sequence_dimension
        int32_t alignment_band_width
        int32_t max_sequences_per_poa
        BandMode band_mode

        BatchConfig(int32_t, int32_t, int32_t, BandMode)
        BatchConfig(int32_t, int32_t, int32_t,
                    int32_t, int32_t, int32_t, BandMode)

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
                                        const BatchConfig&, int16_t, int16_t,
                                        int16_t)
