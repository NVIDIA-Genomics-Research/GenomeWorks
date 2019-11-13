/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdint.h>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace claragenomics
{

using node_id_t = int32_t;
using edge_t    = std::pair<node_id_t, node_id_t>;

/// \struct PairHash
/// Hash function for a pair
struct PairHash
{
public:
    /// \brief Operator overload to define hash function
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& pair) const
    {
        size_t hash_1 = std::hash<T1>()(pair.first);
        size_t hash_2 = std::hash<T2>()(pair.second);
        return hash_1 ^ hash_2;
    }
};

/// \class Graph
/// Object representing a generic graph structure
class Graph
{
public:
    /// \brief Default dtor
    ~Graph() = default;

    /// \brief Add edges to a graph
    ///
    /// \param node_id_from Source node ID
    /// \param node_id_to Sink node ID
    virtual void add_edge(node_id_t node_id_from, node_id_t node_id_to) = 0;

    /// \brief Get a list of adjacent nodes to a given node
    ///
    /// \param node Node for which adjacent nodes are requested
    /// \return Vector of adjacent node IDs
    virtual const std::vector<node_id_t>& get_adjacent_nodes(node_id_t node) const
    {
        auto iter = adjacent_nodes_.find(node);
        if (iter != adjacent_nodes_.end())
        {
            return iter->second;
        }
        else
        {
            return empty_;
        }
    }

    /// \brief List all node IDs in the graph
    ///
    /// \return A vector of node IDs
    virtual const std::vector<node_id_t> get_node_ids() const
    {
        std::vector<node_id_t> nodes;
        for (auto iter : adjacent_nodes_)
        {
            nodes.push_back(iter.first);
        }

        return nodes;
    }

    /// \brief Get a list of all edges in the graph
    ///
    /// \return A vector of edges
    virtual const std::vector<edge_t> get_edges() const
    {
        std::vector<edge_t> edges;
        for (auto iter : edges_)
        {
            edges.push_back(iter);
        }
        return edges;
    }

    /// \brief Add string labels to a node ID
    ///
    /// \param node ID of node
    /// \param label Label to attach to that node ID
    virtual void set_node_label(node_id_t node, const std::string& label)
    {
        node_labels_.insert({node, label});
    }

    /// \brief Get the label associated with a node
    ///
    /// \param node node ID for label query
    /// \return String label for associated node. Returns empty string if
    //          no label is associated or node ID doesn't exist.
    virtual std::string get_node_label(node_id_t node) const
    {
        auto found_node = node_labels_.find(node);
        if (found_node != node_labels_.end())
        {
            return found_node->second;
        }
        else
        {
            return "";
        }
    }

    /// \brief Serialize graph structure to dot format
    ///
    /// \return A string encoding the graph in dot format
    virtual std::string serialize_to_dot() const = 0;

protected:
    Graph() = default;

    /// \brief Check if a directed edge exists in the grph
    ///
    /// \param edge A directed edge
    /// \return Boolean result of check
    bool directed_edge_exists(edge_t edge)
    {
        auto find_edge = edges_.find(edge);
        if (find_edge == edges_.end())
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    /// \brief Update the adjacent nodes based on edge information
    ///
    /// \param edge A directed edge
    void update_adject_nodes(edge_t edge)
    {
        auto find_node = adjacent_nodes_.find(edge.first);
        if (find_node == adjacent_nodes_.end())
        {
            adjacent_nodes_.insert({edge.first, {edge.second}});
        }
        else
        {
            find_node->second.push_back(edge.second);
        }
    }

    /// List of adjacent nodes per node ID
    std::unordered_map<node_id_t, std::vector<node_id_t>> adjacent_nodes_;

    /// All edges in the graph
    std::unordered_set<edge_t, PairHash> edges_;

    /// Label per node
    std::unordered_map<node_id_t, std::string> node_labels_;

    /// An empty list representing no connectivity
    const std::vector<node_id_t> empty_;
};

/// \class DirectedGraph
/// Object representing a directed graph structure
class DirectedGraph : public Graph
{
public:
    DirectedGraph() = default;

    ~DirectedGraph() = default;

    virtual void add_edge(node_id_t node_id_from, node_id_t node_id_to)
    {
        auto edge = edge_t(node_id_from, node_id_to);
        if (!directed_edge_exists(edge))
        {
            edges_.insert(edge);
            update_adject_nodes(edge);
        }
    }

    virtual std::string serialize_to_dot() const
    {
        std::ostringstream dot_str;
        dot_str << "digraph g {\n";
        for (auto iter : adjacent_nodes_)
        {
            node_id_t src    = iter.first;
            auto label_found = node_labels_.find(src);
            if (label_found != node_labels_.end())
            {
                dot_str << src << " [label=\"" << label_found->second << "\"];\n";
            }
            for (node_id_t sink : iter.second)
            {
                dot_str << src << " -> " << sink << "\n";
            }
        }
        dot_str << "}\n";
        return dot_str.str();
    }
};

/// \class UndirectedGraph
/// Object representing an undirected graph structure
class UndirectedGraph : public Graph
{
public:
    UndirectedGraph() = default;

    ~UndirectedGraph() = default;

    virtual void add_edge(node_id_t node_id_from, node_id_t node_id_to)
    {
        auto edge          = edge_t(node_id_from, node_id_to);
        auto edge_reversed = edge_t(node_id_to, node_id_from);
        if (!directed_edge_exists(edge) && !directed_edge_exists(edge_reversed))
        {
            edges_.insert(edge);
            update_adject_nodes(edge);
            update_adject_nodes(edge_reversed);
        }
    }

    virtual std::string serialize_to_dot() const
    {
        std::ostringstream dot_str;
        dot_str << "graph g {\n";

        // Get nodel labels, if any.
        const std::vector<node_id_t> nodes = get_node_ids();
        for (auto node : nodes)
        {
            auto label_found = node_labels_.find(node);
            if (label_found != node_labels_.end())
            {
                dot_str << node << " [label=\"" << label_found->second << "\"];\n";
            }
        }

        // Get edges.
        for (auto iter : edges_)
        {
            dot_str << iter.first << " -- " << iter.second << "\n";
        }

        dot_str << "}\n";
        return dot_str.str();
    }
};

} // namespace claragenomics
