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

#include "cudapoa_structs.cuh"

#include <claraparabricks/genomeworks/utils/cudautils.hpp>

#include <stdio.h>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/**
 * @brief Device function for computing length of path (with heaviest weight) of each node to the head of the POA graph.
 *        The output of this kernel is used in adaptive banding to compute band-width per graph node
 *
 * @param[in] sorted_poa                 Device buffer with sorted graph
 * @param[in] node_count                 Number of graph nodes
 * @param[in] incoming_edge_count        Device buffer with number of incoming edges per node
 * @param[in] local_incoming_edge_count  Device scratch space for maintaining edge counts during distance computation
 * @param[in] incoming_edges             Device buffer with incoming edges per node
 * @param[in] incoming_edge_weights      Device buffer with weights of incoming edges
 * @param[out] node_distance             Device buffer to store computed node distances in the graph
*/
template <typename SizeT>
__device__ void distanceToHeadNode(SizeT* sorted_poa,
                                   SizeT node_count,
                                   uint16_t* incoming_edge_count,
                                   uint16_t* local_incoming_edge_count,
                                   SizeT* incoming_edges,
                                   uint16_t* incoming_edge_weights,
                                   SizeT* node_distance)
{
    // iterate through node IDs and fill out the local incoming edge count
    for (SizeT n = 0; n < node_count; n++)
    {
        local_incoming_edge_count[n] = incoming_edge_count[n];
        // if a node ID has 0 incoming edges, set its distance to 0
        if (local_incoming_edge_count[n] == 0)
        {
            node_distance[n] = 0;
        }
    }

    // the following loop works on a sorted graph
    for (SizeT n = 0; n < node_count; ++n)
    {
        SizeT node = sorted_poa[n];
        // To update distance to head node, loop through all successors and pick the one with heaviest edge
        // then increment the distance of the selected successor node by one and assign it as distance of the current node
        int16_t max_successor_weight = 0;
        SizeT successor_distance     = -1;
        SizeT successor_node         = 0;
        for (uint16_t in_edge = 0; in_edge < incoming_edge_count[node]; ++in_edge)
        {
            if (max_successor_weight < incoming_edge_weights[node * CUDAPOA_MAX_NODE_EDGES + in_edge])
            {
                max_successor_weight = incoming_edge_weights[node * CUDAPOA_MAX_NODE_EDGES + in_edge];
                successor_node       = incoming_edges[node * CUDAPOA_MAX_NODE_EDGES + in_edge];
                successor_distance   = node_distance[successor_node];
            }
        }
        node_distance[node] = successor_distance + 1;
        //printf("> n %3d, sorted_n %3d, distance %3d \n", n, node, successor_distance + 1);
    }
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
