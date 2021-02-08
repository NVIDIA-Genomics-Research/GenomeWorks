/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "../src/cudapoa_structs.cuh" //CUDAPOA_MAX_NODE_EDGES, CUDAPOA_MAX_NODE_ALIGNMENTS

#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp> //get_size

#include <string>
#include <vector>
#include <stdint.h>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

// alias for the 2d vector graph representation
typedef std::vector<std::vector<int16_t>> Int16Vec2D;
typedef std::vector<std::vector<std::vector<uint16_t>>> Uint16Vec3D;
typedef std::vector<std::vector<uint16_t>> Uint16Vec2D;

class BasicGraph
{
public:
    BasicGraph(std::vector<uint8_t> nodes, Int16Vec2D outgoing_edges, Int16Vec2D node_alignments, std::vector<uint16_t> node_coverage_counts, Uint16Vec3D outgoing_edges_coverage = {})
        : nodes_(nodes)
        , outgoing_edges_(outgoing_edges)
        , node_alignments_(node_alignments)
        , node_coverage_counts_(node_coverage_counts)
        , outgoing_edges_coverage_(outgoing_edges_coverage)
    {
        graph_complete_ = true;
        node_count_     = get_size(nodes_);
    }

    BasicGraph(int16_t* outgoing_edges, uint16_t* outgoing_edge_count, int16_t node_count)
    {
        graph_complete_ = false;
        outgoing_edges_ = edges_to_graph(outgoing_edges, outgoing_edge_count, node_count);
    }

    BasicGraph(Int16Vec2D outgoing_edges)
    {
        graph_complete_ = false;
        outgoing_edges_ = outgoing_edges;
        node_count_     = get_size(outgoing_edges);
    }

    BasicGraph(std::vector<uint8_t> nodes, Int16Vec2D outgoing_edges)
        : BasicGraph(outgoing_edges)
    {
        nodes_ = nodes;
    }

    BasicGraph() = delete;

    //fill in the edge-related pointers based on stored graph
    void get_edges(int16_t* incoming_edges, uint16_t* incoming_edge_count,
                   int16_t* outgoing_edges, uint16_t* outgoing_edge_count) const
    {
        int16_t out_node;
        for (int i = 0; i < node_count_; i++)
        {
            outgoing_edge_count[i] = get_size(outgoing_edges_[i]);
            for (int j = 0; j < get_size(outgoing_edges_[i]); j++)
            {
                out_node                                                          = outgoing_edges_[i][j];
                uint16_t in_edge_count                                            = incoming_edge_count[out_node];
                incoming_edge_count[out_node]                                     = in_edge_count + 1;
                incoming_edges[out_node * CUDAPOA_MAX_NODE_EDGES + in_edge_count] = i;
                outgoing_edges[i * CUDAPOA_MAX_NODE_EDGES + j]                    = out_node;
            }
        }
    }
    //fill in the nodes and node_count pointer
    void get_nodes(uint8_t* nodes, int16_t* node_count) const
    {
        for (int i = 0; i < get_size(nodes_); i++)
        {
            nodes[i] = nodes_[i];
        }
        *node_count = node_count_;
    }
    //fill in the node_alignments and node_alignment_count pointers
    void get_node_alignments(int16_t* node_alignments, uint16_t* node_alignment_count) const
    {
        int16_t aligned_node;
        for (int i = 0; i < get_size(node_alignments_); i++)
        {
            for (int j = 0; j < get_size(node_alignments_[i]); j++)
            {
                aligned_node                                         = node_alignments_[i][j];
                node_alignments[i * CUDAPOA_MAX_NODE_ALIGNMENTS + j] = aligned_node;
                node_alignment_count[i]++;
            }
        }
    }
    //fill in node_coverage_counts pointer
    void get_node_coverage_counts(uint16_t* node_coverage_counts) const
    {
        for (int i = 0; i < get_size(node_coverage_counts_); i++)
        {
            node_coverage_counts[i] = node_coverage_counts_[i];
        }
    }

    // convert results from outgoing_edges to Uint16Vec2D graph
    Int16Vec2D edges_to_graph(int16_t* outgoing_edges, uint16_t* outgoing_edge_count, uint16_t node_count)
    {
        Int16Vec2D graph(node_count);
        for (uint16_t i = 0; i < node_count; i++)
        {
            for (uint16_t j = 0; j < outgoing_edge_count[i]; j++)
            {
                graph[i].push_back(outgoing_edges[i * CUDAPOA_MAX_NODE_EDGES + j]);
            }
        }
        return graph;
    }

    void get_outgoing_edges_coverage(uint16_t* outgoing_edges_coverage, uint16_t* outgoing_edges_coverage_count, uint16_t num_sequences) const
    {
        if (outgoing_edges_coverage_.size() == 0)
            return;
        for (uint32_t i = 0; i < outgoing_edges_coverage_.size(); i++) //from_node
        {
            for (uint32_t j = 0; j < outgoing_edges_coverage_[i].size(); j++) //to_node
            {
                uint16_t edge_coverage_count                                  = outgoing_edges_coverage_[i][j].size();
                outgoing_edges_coverage_count[i * CUDAPOA_MAX_NODE_EDGES + j] = edge_coverage_count;
                for (int k = 0; k < edge_coverage_count; k++)
                {
                    outgoing_edges_coverage[i * CUDAPOA_MAX_NODE_EDGES * num_sequences + j * num_sequences + k] = outgoing_edges_coverage_[i][j][k];
                }
            }
        }
    }

    bool is_complete() const
    {
        return graph_complete_;
    }

    bool operator==(const BasicGraph& rhs) const
    {
        return this->outgoing_edges_ == rhs.outgoing_edges_;
    }

    const Int16Vec2D& get_outgoing_edges() const { return outgoing_edges_; }

protected:
    bool graph_complete_;
    std::vector<uint8_t> nodes_;
    Int16Vec2D outgoing_edges_; //this uniquely represents the graph structure; equality of BasicGraph is based on this member.
    Uint16Vec3D outgoing_edges_coverage_;
    Int16Vec2D node_alignments_;
    std::vector<uint16_t> node_coverage_counts_;
    uint16_t node_count_;
};

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
