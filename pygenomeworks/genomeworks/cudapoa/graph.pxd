

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.pair cimport pair
from libcpp.string cimport string
from libc.stdint cimport int32_t
from libcpp.vector cimport vector

# This file declares public structs and API calls
# from the GenomeWorks `graph` utility class.

# Declare structs and APIs from graph.hpp.
cdef extern from "claraparabricks/genomeworks/utils/graph.hpp" namespace "claraparabricks::genomeworks":
    cdef cppclass Graph:
        ctypedef int32_t node_id_t
        ctypedef int32_t edge_weight_t
        ctypedef pair[node_id_t, node_id_t] edge_t

    cdef cppclass DirectedGraph(Graph):
        vector[node_id_t]& get_adjacent_nodes(node_id_t) except +
        vector[node_id_t] get_node_ids() except +
        vector[pair[edge_t, edge_weight_t]] get_edges() except +
        void set_node_label(node_id_t, const string&) except +
        string get_node_label(node_id_t) except +
        void add_edge(node_id_t, node_id_t, edge_weight_t) except +
        string serialize_to_dot() except +
