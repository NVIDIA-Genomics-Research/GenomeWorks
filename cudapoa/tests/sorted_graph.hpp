#include <string>
#include <vector>
#include <stdint.h>
#include "../src/cudapoa_kernels.cuh" //CUDAPOA_MAX_NODE_EDGES, CUDAPOA_MAX_NODE_ALIGNMENTS
#include "basic_graph.hpp"

namespace genomeworks
{

namespace cudapoa
{

class SortedGraph : public BasicGraph
{

public:
    SortedGraph(std::vector<uint8_t> nodes, std::vector<uint16_t> sorted_graph, Uint16Vec2D outgoing_edges)
        : BasicGraph(nodes, outgoing_edges), sorted_graph_(sorted_graph)
    {
        // do nothing for now
    }

    SortedGraph(std::vector<uint8_t> nodes, std::vector<uint16_t> sorted_graph,
                Uint16Vec2D node_alignments, std::vector<uint16_t> node_coverage_counts,
                Uint16Vec2D outgoing_edges, Uint16Vec3D outgoing_edges_coverage = {})
        : BasicGraph(nodes, outgoing_edges, node_alignments, node_coverage_counts, outgoing_edges_coverage), sorted_graph_(sorted_graph)
    {
        // do nothing for now
    }
    SortedGraph() = delete;

    void get_node_id_to_pos(uint16_t* node_id_to_pos) const
    {
        for (size_t pos = 0; pos < sorted_graph_.size(); pos++)
        {
            uint16_t id                = sorted_graph_[pos];
            node_id_to_pos[(size_t)id] = (uint16_t)pos;
        }
    }

    void get_sorted_graph(uint16_t* graph) const
    {
        for (int i = 0; i < sorted_graph_.size(); i++)
        {
            graph[i] = sorted_graph_[i];
        }
    }

protected:
    std::vector<uint16_t> sorted_graph_;
};

} // namespace cudapoa

} // namespace genomeworks
