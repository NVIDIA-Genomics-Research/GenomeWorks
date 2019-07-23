# distutils: language = c++

from cython.operator cimport dereference as deref
from claragenomics.bindings.cudapoa cimport StatusType, OutputType, Batch, create_batch
from claragenomics.bindings.cuda import CudaStream
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libc.stdint cimport uint16_t

cdef class CudaPoaBatch:
    """
    Python API for CUDA accelerated partial order alignment algorithm.
    """
    cdef unique_ptr[Batch] batch

    def __cinit__(
            self,
            max_poas,
            max_sequences_per_poa,
            stream=None,
            device_id=0,
            output_mask=consensus,
            gap_score=-8,
            mismatch_score=-6,
            match_score=8,
            cuda_banded_alignment=False):
        """
        Construct a CUDAPOA Batch object to run CUDA accelerated
        partial order alignment across all windows in the batch.

        Args:
            max_poas : Maximum number of partial order alignments to
                       to perform in batch.
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
        if (stream == None):
            temp_stream = NULL
        elif (not isinstance(stream, CudaStream)):
            raise RuntimeError("Type for stream option must be CudaStream")
        else:
            st = stream.get_stream()
            temp_stream = <_Stream>st

        self.batch = create_batch(max_poas,
                max_sequences_per_poa, 
                temp_stream, 
                device_id,
                output_mask,
                gap_score,
                mismatch_score,
                match_score,
                cuda_banded_alignment)

    def add_poas(self, poas):
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
        if (len(poas) < 1):
            raise RuntimeError("At least one poa must be added to CudaPoaBatch")
        if (not isinstance(poas[0], list)):
            poas = [poas]
        cdef bytes py_bytes
        cdef char* c_string
        for poa in poas:
            status = deref(self.batch).add_poa()
            if status != success:
                raise RuntimeError("Could not add new POA: " + str(status))
            for seq in poa:
                py_bytes = seq.encode()
                c_string = py_bytes
                status = deref(self.batch).add_seq_to_poa(c_string, NULL, len(seq))
                if status != success:
                    raise RuntimeError("Could not add new sequence to poa")

    def get_total_poas(self):
        """
        Get total number of POA groups added to batch.

        Returns:
            Number of POA groups added to batch.
        """
        return deref(self.batch).get_total_poas()

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
        cdef vector[StatusType] status
        error = deref(self.batch).get_msa(msa, status)
        if error == output_type_unavailable:
            raise RuntimeError("Output type not requested during batch initialization")
        return msa

    def get_consensus(self):
        """
        Get the consensus for each POA group.

        Returns:
            A list with consensus string for each group.
        """
        cdef vector[string] consensus
        cdef vector[vector[uint16_t]] coverage
        cdef vector[StatusType] status
        error = deref(self.batch).get_consensus(consensus, coverage, status)
        if error == output_type_unavailable:
            raise RuntimeError("Output type not requested during batch initialization")
        return consensus
