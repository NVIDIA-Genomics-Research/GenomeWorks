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

"""CUDAPOA binding module."""

import networkx as nx

import genomeworks.cuda as cuda

from cython.operator cimport dereference as deref
from libc.stdint cimport uint16_t, int8_t, int32_t
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

from genomeworks.cuda.cuda_runtime_api cimport _Stream
from genomeworks.cudapoa.graph cimport DirectedGraph
cimport genomeworks.cudapoa.cudapoa as cudapoa


def status_to_str(status):
    """Convert status to their string representations."""
    if status == cudapoa.success:
        return "success"
    elif status == cudapoa.exceeded_maximum_poas:
        return "exceeded_maximum_poas"
    elif status == cudapoa.exceeded_maximum_sequence_size:
        return "exceeded_maximum_sequence_size"
    elif status == cudapoa.exceeded_maximum_sequences_per_poa:
        return "exceeded_maximum_sequences_per_poa"
    elif status == cudapoa.node_count_exceeded_maximum_graph_size:
        return "node_count_exceeded_maximum_graph_size"
    elif status == cudapoa.edge_count_exceeded_maximum_graph_size:
        return "edge_count_exceeded_maximum_graph_size"
    elif status == cudapoa.seq_len_exceeded_maximum_nodes_per_window:
        return "seq_len_exceeded_maximum_nodes_per_window"
    elif status == cudapoa.loop_count_exceeded_upper_bound:
        return "loop_count_exceeded_upper_bound"
    elif status == cudapoa.output_type_unavailable:
        return "output_type_unavailable"
    elif status == cudapoa.generic_error:
        return "generic_error"
    else:
        raise RuntimeError("Unknown error status : " + status)


cdef class CudaPoaBatch:
    """Python API for CUDA-accelerated partial order alignment algorithm."""
    cdef unique_ptr[cudapoa.Batch] batch
    cdef unique_ptr[cudapoa.BatchConfig] batch_size

    def __cinit__(
            self,
            max_sequences_per_poa,
            max_sequence_size,
            max_gpu_mem,
            output_type="consensus",
            band_mode="adaptive_band",
            device_id=0,
            stream=None,
            gap_score=-8,
            mismatch_score=-6,
            match_score=8,
            alignment_band_width=256,
            max_consensus_size=None,
            max_nodes_per_graph=None,
            matrix_sequence_dimension=None,
            *args,
            **kwargs):
        """Construct a CUDAPOA Batch object to run CUDA-accelerated
        partial order alignment across all windows in the batch.

        Args:
            max_sequences_per_poa : Maximum number of sequences per POA
            max_sequence_size : Maximum number of elements in a sequence
            max_gpu_mem : Maximum GPU memory to use for this batch
            output_type : Types of outputs to generate (consensus, msa)
            band_mode : Operation mode (full_band, static_band, adaptive_band)
            device_id : ID of GPU device to use
            stream : CudaStream to use for GPU execution
            gap_score : Penalty for gaps
            mismatch_score : Penalty for mismatches
            match_score : Reward for match
            alignment_band_width : Band-width size if using banded alignment
            max_consensus_size : Maximum size of final consensus
            max_nodes_per_graph : Maximum number of nodes in a graph, 1 graph per window
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

        if (output_type == "consensus"):
            output_mask = cudapoa.consensus
        elif (output_type == "msa"):
            output_mask = cudapoa.msa
        else:
            raise RuntimeError("Unknown output_type provided. Must be consensus/msa.")

        # Since cython make_unique doesn't accept python objects, need to
        # store it in a cdef and then pass into the make unique call
        cdef int32_t mx_seq_sz = max_sequence_size
        cdef int32_t band_width_sz = alignment_band_width
        cdef int32_t mx_seq_per_poa = max_sequences_per_poa
        cdef int32_t mx_consensus_sz = \
            2 * max_sequence_size if max_consensus_size is None else max_consensus_size
        cdef int32_t mx_nodes_per_w
        cdef int32_t matrix_seq_dim
        cdef BandMode batch_band_mode
        if (band_mode == "full_band"):
            batch_band_mode = BandMode.full_band
            mx_nodes_per_w = 3 * max_sequence_size if max_nodes_per_graph is None else max_nodes_per_graph
            matrix_seq_dim = max_sequence_size if matrix_sequence_dimension is None else matrix_sequence_dimension
        elif (band_mode == "static_band"):
            batch_band_mode = BandMode.static_band
            mx_nodes_per_w = 4 * max_sequence_size if max_nodes_per_graph is None else max_nodes_per_graph
            matrix_seq_dim = ((alignment_band_width + 8) if matrix_sequence_dimension is None
                              else matrix_sequence_dimension)
        elif (band_mode == "adaptive_band"):
            batch_band_mode = BandMode.adaptive_band
            mx_nodes_per_w = 4 * max_sequence_size if max_nodes_per_graph is None else max_nodes_per_graph
            matrix_seq_dim = (2*(alignment_band_width + 8) if matrix_sequence_dimension is None
                              else matrix_sequence_dimension)
        else:
            raise RuntimeError("Unknown band_mode provided. Must be full_band/static_band/adaptive_band.")

        self.batch_size = make_unique[cudapoa.BatchConfig](
            mx_seq_sz, mx_consensus_sz, mx_nodes_per_w, band_width_sz, mx_seq_per_poa, matrix_seq_dim, batch_band_mode)

        self.batch = cudapoa.create_batch(
            device_id,
            temp_stream,
            max_gpu_mem,
            output_mask,
            deref(self.batch_size),
            gap_score,
            mismatch_score,
            match_score)

    def __init__(
            self,
            max_sequences_per_poa,
            max_sequence_size,
            max_gpu_mem,
            output_type="consensus",
            band_mode="static_band",
            device_id=0,
            stream=None,
            gap_score=-8,
            mismatch_score=-6,
            match_score=8,
            alignment_band_width=256,
            max_consensus_size=None,
            max_nodes_per_graph=None,
            *args,
            **kwargs):
        """Dummy implementation of __init__ function to allow
        for Python subclassing.
        """
        pass

    def add_poa_group(self, poa):
        """Set the POA groups to run alignment on.

        Args:
            poas : List of POA groups. Each group is a list of
                   sequences.
                   e.g.
                   [["ACTG", "ATCG"], <--- Group 1
                   ["GCTA", "GACT", "ACGTC"]] <--- Group 2
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
        byte_list = []  # To store byte array of POA sequences
        for seq in poa:
            byte_list.append(seq.encode('utf-8'))
            c_string = byte_list[-1]
            entry.seq = c_string
            entry.weights = NULL
            entry.length = len(seq)
            poa_group.push_back(entry)
        status = deref(self.batch).add_poa_group(seq_status, poa_group)
        return (status, seq_status)

    @property
    def total_poas(self):
        """Get total number of POA groups added to batch.

        Returns:
            Number of POA groups added to batch.
        """
        return deref(self.batch).get_total_poas()

    @property
    def batch_id(self):
        """Get the batch ID of the cudapoa Batch object.

        Returns:
            Batch ID.
        """
        return deref(self.batch).batch_id()

    def generate_poa(self):
        """Run asynchronous partial order alignment on all POA groups
        in batch.
        """
        deref(self.batch).generate_poa()

    def get_msa(self):
        """Get the multi-sequence alignment for each POA group.

        Returns:
            A tuple where
            - first element is a list with MSAs for each POA group, with each MSA
            represented as a list of alignments.
            - second element is status of MSA generation for each group
        """
        cdef vector[vector[string]] msa
        cdef vector[cudapoa.StatusType] status
        error = deref(self.batch).get_msa(msa, status)
        if error == cudapoa.output_type_unavailable:
            raise RuntimeError("Output type not requested during batch initialization")
        return (msa, status)

    def get_consensus(self):
        """Get the consensus for each POA group.

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
        if error == cudapoa.output_type_unavailable:
            raise RuntimeError("Output type not requested during batch initialization")
        decoded_consensus = [c.decode('utf-8') for c in consensus]
        return (decoded_consensus, coverage, status)

    def get_graphs(self):
        """Get the POA graph for each POA group.

        Returns:
            A tuple where
            - first element is a networkx graph for each POA group
            - second element is status of MSA generation for each group
        """
        cdef vector[DirectedGraph] graphs
        cdef vector[cudapoa.StatusType] status
        cdef vector[pair[DirectedGraph.edge_t, DirectedGraph.edge_weight_t]] edges
        cdef DirectedGraph* graph
        cdef DirectedGraph.edge_t edge
        cdef DirectedGraph.edge_weight_t weight

        # Get the graphs from batch object.
        deref(self.batch).get_graphs(graphs, status)

        nx_digraphs = []
        for g in range(graphs.size()):
            graph = &graphs[g]
            edges = deref(graph).get_edges()
            nx_digraph = nx.DiGraph()
            for e in range(edges.size()):
                edge = edges[e].first
                weight = edges[e].second
                nx_digraph.add_edge(edge.first,
                                    edge.second,
                                    weight=weight)
            attributes = {}
            for n in nx_digraph.nodes:
                attributes[n] = {'label': deref(graph).get_node_label(n).decode('utf-8')}
            nx.set_node_attributes(nx_digraph, attributes)
            nx_digraphs.append(nx_digraph)
        return (nx_digraphs, status)

    def reset(self):
        """Reset the batch object. Involves deleting all windows previously
        assigned to batch object.
        """
        deref(self.batch).reset()
