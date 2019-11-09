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

/// \struct pair_hasher
struct PairHash
{
public:
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& pair) const
    {
        size_t hash_1 = std::hash<T1>()(pair.first);
        size_t hash_2 = std::hash<T2>()(pair.second);
        return hash_1 ^ hash_2;
    }
};

/// \class DirectedGraph
/// Object representing a graph structure
class DirectedGraph
{
public:
    DirectedGraph() = default;

    ~DirectedGraph() = default;

    virtual void add_edge(node_id_t node_id_from, node_id_t node_id_to)
    {
        auto edge      = std::pair<node_id_t, node_id_t>(node_id_from, node_id_to);
        auto find_edge = edges_.find(edge);
        if (find_edge == edges_.end())
        {
            edges_.insert(edge);
            auto find_node = adjacent_nodes_.find(node_id_from);
            if (find_node == adjacent_nodes_.end())
            {
                adjacent_nodes_.insert({node_id_from, {node_id_to}});
            }
            else
            {
                find_node->second.push_back(node_id_to);
            }
        }
    }

    virtual void add_label(node_id_t node, const std::string& label)
    {
        node_labels_.insert({node, label});
    }

    virtual const std::vector<node_id_t>& get_adjacent_nodes(node_id_t node)
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

    virtual const std::vector<node_id_t> get_node_ids()
    {
        std::vector<node_id_t> nodes;
        for (auto iter : adjacent_nodes_)
        {
            nodes.push_back(iter.first);
        }

        return nodes;
    }

    virtual std::string get_node_label(node_id_t node)
    {
        auto found_node = node_labels_.find(node);
        if (found_node != node_labels_.end())
        {
            return found_node->second;
        }
        else
        {
            throw std::runtime_error("No node found with given ID");
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
            dot_str << src << " [label=\"" << label_found->second << "\"];\n";
            for (node_id_t sink : iter.second)
            {
                dot_str << src << " -> " << sink << "\n";
            }
        }
        dot_str << "\n";
        return dot_str.str();
    }

private:
    std::unordered_map<node_id_t, std::vector<node_id_t>> adjacent_nodes_;
    std::unordered_set<std::pair<node_id_t, node_id_t>, PairHash> edges_;
    std::unordered_map<node_id_t, std::string> node_labels_;
    const std::vector<node_id_t> empty_;
};

} // namespace claragenomics
