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

from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int32_t, int64_t
from libcpp.memory cimport unique_ptr, shared_ptr

from genomeworks.cuda.cuda_runtime_api cimport _Stream

# This file declared public structs and API calls
# from the GenomeWorks `cudaaligner` module.

# Declare structs and APIs from cudaaligner.hpp
cdef extern from "claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp" \
        namespace "claraparabricks::genomeworks::cudaaligner":
    cdef enum StatusType:
        success = 0,
        uninitialized
        exceeded_max_alignments
        exceeded_max_length
        exceeded_max_alignment_difference
        generic_error

    cdef enum AlignmentType:
        global_alignment = 0
        unset

    cdef enum AlignmentState:
        match = 0
        mismatch
        insertion  # Absent in query, present in target
        deletion   # Present in query, absent in target

    cdef StatusType Init()

# Declare structs and APIs from alignment.hpp
cdef extern from "claraparabricks/genomeworks/cudaaligner/alignment.hpp" \
        namespace "claraparabricks::genomeworks::cudaaligner":
    ctypedef struct  FormattedAlignment:
        string query
        string pairing
        string target

    cdef cppclass Alignment:
        string get_query_sequence() except +
        string get_target_sequence() except +
        string convert_to_cigar() except +
        AlignmentType get_alignment_type() except +
        StatusType get_status() except +
        vector[AlignmentState] get_alignment() except +
        FormattedAlignment format_alignment() except +

# Declare structs and APIs from aligner.hpp
cdef extern from "claraparabricks/genomeworks/cudaaligner/aligner.hpp" \
        namespace "claraparabricks::genomeworks::cudaaligner":
    cdef cppclass Aligner:
        StatusType align_all() except +
        StatusType sync_alignments() except +
        StatusType add_alignment(const char*, int32_t, const char*, int32_t) except +
        vector[shared_ptr[Alignment]] get_alignments() except +
        void reset() except +

    unique_ptr[Aligner] create_aligner(int32_t, int32_t, int32_t, AlignmentType, _Stream, int32_t, int64_t)
