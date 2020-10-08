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

"""Bindings for CUDAALIGNER."""


from cython.operator cimport dereference as deref
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.string cimport string
from libc.stdint cimport uint16_t

from cpython.ref cimport Py_INCREF, Py_DECREF
from genomeworks.cuda.cuda_runtime_api cimport _Stream
from genomeworks.cuda.cuda cimport CudaStream
cimport genomeworks.cudaaligner.cudaaligner as cudaaligner


def status_to_str(status):
    """Convert status to their string representations.
    """
    if status == cudaaligner.success:
        return "success"
    elif status == cudaaligner.uninitialized:
        return "uninitialized"
    elif status == cudaaligner.exceeded_max_alignments:
        return "exceeded_max_alignments"
    elif status == cudaaligner.exceeded_max_length:
        return "exceeded_max_length"
    elif status == cudaaligner.exceeded_max_alignment_difference:
        return "exceeded_max_alignment_difference"
    elif status == cudaaligner.generic_error:
        return "generic_error"
    else:
        raise RuntimeError("Unknown error status : " + status)


class CudaAlignment:
    """Class encompassing an Alignment between two sequences.
    """
    def __init__(self,
                 query,
                 target,
                 cigar,
                 alignment_type,
                 status,
                 alignment,
                 format_alignment):
        """Construct an alignment object based on alignment information between two
        sequences.

        Args:
            query - Query string
            target - Target string
            cigar - CIGAR string for operations needed to go from query -> target
            alignment_type - cudaaligner.AlignmentType enum representing type of alignment
            status - StatusType enum from CudaAlignerBatch call for this alignment
            alignment - list of edit operations to align query and target
            format_alignment - a pair of strings with the formatted alignment
        """
        self.query = query
        self.target = target
        self.cigar = cigar
        self.alignment_type = self._alignment_type_str(alignment_type)
        self.status = status
        self.alignment = [self._alignment_state_enum_str(s) for s in alignment]
        self.format_alignment = format_alignment

    @staticmethod
    def _alignment_type_str(t):
        """Convert alignment type enum to string.

        Args:
            t - alignment type

        Returns: Alignment type string
        """
        if t == cudaaligner.global_alignment:
            return "global"
        else:
            raise RuntimeError("Unknown alignment type encountered: " + t)

    @staticmethod
    def _alignment_state_enum_str(s):
        """Convert alignment state enum to string.

        Args:
            s - alignment state

        Returns:
            Alignment state string
        """
        if s == cudaaligner.match:
            return 'm'
        elif s == cudaaligner.mismatch:
            return 'mm'
        elif s == cudaaligner.insertion:
            return 'i'
        elif s == cudaaligner.deletion:
            return 'd'
        else:
            raise RuntimeError("Unknown alignment state encountered: " + s)

    def __str__(self):
        """Print formatted alignment.
        """
        return "{}\n{}\n{}\n".format(self.format_alignment[0], self.format_alignment[1], self.format_alignment[2])


cdef class CudaAlignerBatch:
    """Python API for CUDA-accelerated sequence to sequence alignment.
    """
    cdef unique_ptr[cudaaligner.Aligner] aligner
    cdef public CudaStream stream

    def __cinit__(
            self,
            max_query_length,
            max_target_length,
            max_alignments,
            alignment_type="global",
            stream=None,
            device_id=0,
            max_device_memory_allocator_caching_size=-1,
            *args,
            **kwargs):
        """Construct a CudaAligner object to run CUDA-accelerated sequence
        to sequence alignment across all pairs in a batch.

        Args:
            max_query_length - Max length of query string
            max_target_length - Max length of target string
            max_alignments - Maximum number of alignments to perform
            alignment_type - Type of alignment (only global supported right now)
            stream - CUDA stream for running kernel
            device_id - GPU device to use for running kernels
            max_device_memory_allocator_caching_size - Maximum amount of device memory to use for cached memory
            allocations the cudaaligner instance. max_device_memory_allocator_caching_size = -1 (default) means
            all available device memory.
        """
        cdef size_t st
        cdef _Stream temp_stream
        if (stream is None):
            temp_stream = NULL
        elif (not isinstance(stream, CudaStream)):
            raise RuntimeError("Type for stream option must be CudaStream")
        else:
            st = stream.stream
            temp_stream = <_Stream>st
        # keep a reference to the stream, such that it gets destroyed after the aligner.
        self.stream = stream
        # Increasing ref count of CudaStream object to ensure it doesn't get garbage
        # collected before CudaAlignerBatch object.
        # NOTE: Ideally this is taken care of by just storing the reference
        # in the line above, but that doesn't seem to be persistent.
        Py_INCREF(stream)

        cdef cudaaligner.AlignmentType alignment_type_enum
        if (alignment_type == "global"):
            alignment_type_enum = cudaaligner.global_alignment
        else:
            raise RuntimeError("Unknown alignment_type provided. Must be global.")

        self.aligner = cudaaligner.create_aligner(
            max_query_length,
            max_target_length,
            max_alignments,
            alignment_type_enum,
            temp_stream,
            device_id,
            max_device_memory_allocator_caching_size)

    def __init__(
            self,
            max_query_length,
            max_target_length,
            max_alignments,
            alignment_type="global",
            stream=None,
            device_id=0,
            max_device_memory_allocator_caching_size=-1,
            *args,
            **kwargs):
        """Dummy implementation of __init__ function to allow
        for Python subclassing.
        """
        pass

    def add_alignment(self, query, target):
        """Add new pair of sequences to the batch for alignment.
        The characters in the string must be from the set [ACGT] for
        correct alignment results.

        Args:
            query - query string
            target - target string
        """
        encoded_query = query.encode('utf-8')
        encoded_target = target.encode('utf-8')
        status = deref(self.aligner).add_alignment(encoded_query, len(query),
                                                   encoded_target, len(target))

        return status

    def align_all(self):
        """Initiate CUDA-accelerated alignment on the batch.
        """
        deref(self.aligner).align_all()

    def get_alignments(self):
        """Retrieve the results of all alignments in the batch.

        Returns:
        A list of CudaAlignment objects, each of which holds details of the alignment in the same
        order as they were inserted into the batch.
        """
        # Declare cdef types
        cdef vector[shared_ptr[cudaaligner.Alignment]] res = deref(self.aligner).get_alignments()
        cdef size_t num_alignments = res.size()
        cdef vector[cudaaligner.AlignmentState] alignment_state
        cdef cudaaligner.FormattedAlignment formatted_alignment

        # First sync all the alignments since the align call is asynchronous.
        deref(self.aligner).sync_alignments()

        alignments = []

        for i in range(num_alignments):
            # Get alignment state
            alignment_state = deref(res[i]).get_alignment()
            state = [alignment_state[j] for j in range(alignment_state.size())]

            # Get formatted alignment
            formatted_alignment = deref(res[i]).format_alignment()
            format_alignment = [formatted_alignment.query.decode('utf-8'),
                                formatted_alignment.pairing.decode('utf-8'),
                                formatted_alignment.target.decode('utf-8')]

            # Get other string outputs
            query = deref(res[i]).get_query_sequence().decode('utf-8')
            target = deref(res[i]).get_target_sequence().decode('utf-8')
            cigar = deref(res[i]).convert_to_cigar().decode('utf-8')
            alignment_type = deref(res[i]).get_alignment_type()
            status = deref(res[i]).get_status()

            # Create Alignment object
            alignments.append(CudaAlignment(
                query,
                target,
                cigar,
                alignment_type,
                status,
                state,
                format_alignment))
        return alignments

    def __dealloc__(self):
        self.aligner.reset()
        # Decreasing ref count for CudaStream object.
        Py_DECREF(self.stream)

    def reset(self):
        """Reset the contents of the batch so the same GPU memory can be used to
        align a new set of sequences.
        """
        deref(self.aligner).reset()
