/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "basic_graph.hpp"
#include "../src/cudapoa_kernels.cuh" //CUDAPOA_MAX_NODE_EDGES, CUDAPOA_MAX_NODE_ALIGNMENTS

#include <claragenomics/utils/signed_integer_utils.hpp> // get_size

#include <string>
#include <vector>
#include <stdint.h>

namespace claragenomics
{

namespace cudapoa
{

class SortedGraph : public BasicGraph
{

public:
    SortedGraph(std::vector<uint8_t> nodes, std::vector<SizeT> sorted_graph, SizeTVec2D outgoing_edges)
        : BasicGraph(nodes, outgoing_edges)
        , sorted_graph_(sorted_graph)
    {
        // do nothing for now
    }

    SortedGraph(std::vector<uint8_t> nodes, std::vector<SizeT> sorted_graph,
                SizeTVec2D node_alignments, std::vector<uint16_t> node_coverage_counts,
                SizeTVec2D outgoing_edges, Uint16Vec3D outgoing_edges_coverage = {})
        : BasicGraph(nodes, outgoing_edges, node_alignments, node_coverage_counts, outgoing_edges_coverage)
        , sorted_graph_(sorted_graph)
    {
        // do nothing for now
    }
    SortedGraph() = delete;

    void get_node_id_to_pos(SizeT* node_id_to_pos) const
    {
        for (int32_t pos = 0; pos < get_size(sorted_graph_); pos++)
        {
            int32_t id         = sorted_graph_[pos];
            node_id_to_pos[id] = static_cast<SizeT>(pos);
        }
    }

    void get_sorted_graph(SizeT* graph) const
    {
        for (int i = 0; i < get_size(sorted_graph_); i++)
        {
            graph[i] = sorted_graph_[i];
        }
    }

protected:
    std::vector<SizeT> sorted_graph_;
};

} // namespace cudapoa

} // namespace claragenomics
