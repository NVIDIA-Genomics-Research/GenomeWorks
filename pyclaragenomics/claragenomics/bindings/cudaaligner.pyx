#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# distutils: language = c++

from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport uint16_t

from claragenomics.bindings cimport cudaaligner
from claragenomics.bindings import cuda

class CudaAlignment:

    def __init__(self,
            query,
            target,
            cigar,
            alignment_type,
            status,
            alignment,
            format_alignment):

        self.query = query
        self.target = target
        self.cigar = cigar
        self.alignment_type = alignment_type
        self.status = status
        self.alignment = alignment
        self.format_alignment = format_alignment

cdef class CudaAlignerBatch:
    cdef unique_ptr[cudaaligner.Aligner] aligner

    # TODO: add args, kwargs
    def __cinit__(
            self,
            max_query_length,
            max_target_length,
            max_alignments,
            alignment_type,
            stream,
            device_id,
            *args,
            **kwargs):
        cdef size_t st
        cdef _Stream temp_stream
        if (stream is None):
            temp_stream = NULL
        elif (not isinstance(stream, cuda.CudaStream)):
            raise RuntimeError("Type for stream option must be CudaStream")
        else:
            st = stream.stream
            temp_stream = <_Stream>st

        cdef cudaaligner.AlignmentType alignment_type_enum
        if (alignment_type == "global"):
            alignment_type_enum = global_alignment
        else:
            raise RuntimeError("Unknown alignment_type provided. Must be global.")

        self.aligner = cudaaligner.create_aligner(
                max_query_length,
                max_target_length,
                max_alignments,
                alignment_type_enum,
                temp_stream,
                device_id)

    #TODO: add args, kwargs
    def __init__(
            self,
            max_query_length,
            max_target_length,
            max_alignments,
            alignment_type,
            stream,
            device_id,
            *args,
            **kwargs):
        pass

    def add_alignment(self, align_pair):
        query = align_pair[0]
        target = align_pair[1]

        encoded_query = query.encode('utf-8')
        encoded_target = target.encode('utf-8')
        status = deref(self.aligner).add_alignment(encoded_query, len(query),
                                                   encoded_target, len(target))
        if status != success:
            raise RuntimeError("Could not add alignment: Error code " + str(status))

    def align_all(self):
        deref(self.aligner).align_all()

    def sync_alignments(self):
        deref(self.aligner).sync_alignments()

    def get_alignments(self):
        cdef vector[shared_ptr[cudaaligner.Alignment]] res = deref(self.aligner).get_alignments()
        cdef size_t num_alignments = res.size()
        cdef vector[AlignmentState] al_state
        cdef FormattedAlignment f
        alignments = []
        for i in range(num_alignments):
            al_state = deref(res[i]).get_alignment()
            state = [al_state[j] for j in range(al_state.size())]
            f = deref(res[i]).format_alignment()
            format_alignment = [f.first.decode('utf-8'), f.second.decode('utf-8')]
            alignments.append(CudaAlignment(
                deref(res[i]).get_query_sequence().decode('utf-8'),
                deref(res[i]).get_target_sequence().decode('utf-8'),
                deref(res[i]).convert_to_cigar().decode('utf-8'),
                deref(res[i]).get_alignment_type(),
                deref(res[i]).get_status(),
                state,
                format_alignment
                )
                )
        return alignments
