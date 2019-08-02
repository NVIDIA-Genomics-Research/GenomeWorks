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

from claragenomics.bindings cimport cudapoa
from claragenomics.bindings import cuda

cdef class CudaPoaBatch:
    """
    Python API for CUDA-accelerated partial order alignment algorithm.
    """
    cdef unique_ptr[cudapoa.Batch] batch

    def __cinit__(
            self,
            max_sequences_per_poa,
            gpu_mem,
            device_id=0,
            stream=None,
            output_mask=consensus,
            gap_score=-8,
            mismatch_score=-6,
            match_score=8,
            cuda_banded_alignment=False,
            *args,
            **kwargs):
        """
        Construct a CUDAPOA Batch object to run CUDA-accelerated
        partial order alignment across all windows in the batch.

        Args:
            max_sequences_per_poa : Maximum number of sequences per POA
            stream : CudaStream to use for GPU execution
            device_id : ID of GPU device to use
            output_mask : Types of outputs to generate (consensus, msa)
            gap_score : Penalty for gaps
            mismatch_score : Penalty for mismatches
            match_score : Reward for match
            cuda_banded_alignment : Run POA using banded alignment
        """
        cdef size_t st
        cdef _Stream temp_stream
        if (stream is None):
            temp_stream = NULL
        elif (not isinstance(stream, cuda.CudaStream)):
            raise RuntimeError("Type for stream option must be CudaStream")
        else:
            st = stream.stream
            temp_stream = <_Stream>st

        self.batch = cudapoa.create_batch(
                max_sequences_per_poa,
                device_id,
                temp_stream,
                gpu_mem,
                output_mask,
                gap_score,
                mismatch_score,
                match_score,
                cuda_banded_alignment)

    def __init__(
            self,
            max_sequences_per_poa,
            gpu_mem,
            device_id=0,
            stream=None,
            output_mask=consensus,
            gap_score=-8,
            mismatch_score=-6,
            match_score=8,
            cuda_banded_alignment=False,
            *args,
            **kwargs):
        """
        Dummy implementation of __init__ function to allow
        for Python subclassing.
        """
        pass

    def add_poa_group(self, poa):
        """
        Set the POA groups to run alignment on.

        Args:
            poas : List of POA groups. Each group is a list of
                   sequences.
                   e.g. [["ACTG", "ATCG"], <--- Group 1
                         ["GCTA", "GACT", "ACGTC"] <--- Group 2
                        ]
                   Throws exception if error is encountered while
                   adding POA groups.
        """
        if (not isinstance(poa, list)):
            poas = [poa]
        if (len(poa) < 1):
            raise RuntimeError("At least one sequence must be present in POA group")
        cdef char* c_string
        cdef cudapoa.Group poa_group
        cdef cudapoa.Entry entry
        cdef vector[cudapoa.StatusType] seq_status
        byte_list = [] # To store byte array of POA sequences
        for seq in poa:
            byte_list.append(seq.encode('utf-8'))
            c_string = byte_list[-1]
            entry.seq = c_string
            entry.weights = NULL
            entry.length = len(seq)
            poa_group.push_back(entry)
        status = deref(self.batch).add_poa_group(seq_status, poa_group)
        if status != success:
            raise RuntimeError("Could not add POA group: " + str(status))
        return (status, seq_status)

    @property
    def total_poas(self):
        """
        Get total number of POA groups added to batch.

        Returns:
            Number of POA groups added to batch.
        """
        return deref(self.batch).get_total_poas()

    @property
    def batch_id(self):
        """
        Get the batch ID of the cudapoa Batch object.

        Returns:
            Batch ID.
        """
        return deref(self.batch).batch_id()

    def generate_poa(self):
        """
        Run asynchronous partial order alignment on all POA groups
        in batch.
        """
        deref(self.batch).generate_poa()

    def get_msa(self):
        """
        Get the multi-sequence alignment for each POA group.

        Returns:
            A list with MSAs for each POA group, with each MSA
            represented as a list of alignments.
        """
        cdef vector[vector[string]] msa
        cdef vector[cudapoa.StatusType] status
        error = deref(self.batch).get_msa(msa, status)
        if error == output_type_unavailable:
            raise RuntimeError("Output type not requested during batch initialization")
        return msa

    def get_consensus(self):
        """
        Get the consensus for each POA group.

        Returns:
            A tuple where
            - first element is list with consensus string for each group
            - second element is a list of per base coverages for each consensus
            - third element is status of consensus generation for each group
        """
        cdef vector[string] consensus
        cdef vector[vector[uint16_t]] coverage
        cdef vector[cudapoa.StatusType] status
        error = deref(self.batch).get_consensus(consensus, coverage, status)
        if error == output_type_unavailable:
            raise RuntimeError("Output type not requested during batch initialization")
        decoded_consensus = [c.decode('utf-8') for c in consensus]
        return (decoded_consensus, coverage, status)

    def reset(self):
        """
        Reset the batch object. Involves deleting all windows previously
        assigned to batch object.
        """
        deref(self.batch).reset()
