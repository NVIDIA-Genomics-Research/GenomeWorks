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
 * @brief Device function for running topoligical sort on graph.
 *
 * @param[out] sorted_poa                Device buffer with sorted graph
 * @param[out] sorted_poa_node_map       Device scratch space for mapping node ID to position in graph
 * @param[in] node_count                 Number of nodes graph
 * @param[in] incoming_edge_count        Device buffer with number of incoming edges per node
 * @param[in] outgoing_edges             Device buffer with outgoing edges per node
 * @param[in] outgoing_edge_count        Device buffer with number of outgoing edges per node
 * @param[in] local_incoming_edge_count  Device scratch space for maintaining edge counts during topological sort
 */
template <typename SizeT>
__device__ void topologicalSortDeviceUtil(SizeT* sorted_poa,
                                          SizeT* sorted_poa_node_map,
                                          int32_t node_count,
                                          uint16_t* incoming_edge_count,
                                          SizeT* outgoing_edges,
                                          uint16_t* outgoing_edge_count,
                                          uint16_t* local_incoming_edge_count)
{
    //printf("Running top sort\n");
    // Clear the incoming edge count for each node.
    //__shared__ int16_t local_incoming_edge_count[CUDAPOA_MAX_NODES_PER_WINDOW];
    //memset(local_incoming_edge_count, -1, CUDAPOA_MAX_NODES_PER_WINDOW);
    int32_t sorted_poa_position = 0;

    // Iterate through node IDs (since nodes are from 0
    // through node_count -1, a simple loop works) and fill
    // out the incoming edge count.
    for (int32_t n = 0; n < node_count; n++)
    {
        local_incoming_edge_count[n] = incoming_edge_count[n];
        // If we find a node ID has 0 incoming edges, add it to sorted nodes list.
        if (local_incoming_edge_count[n] == 0)
        {
            sorted_poa_node_map[n]            = sorted_poa_position;
            sorted_poa[sorted_poa_position++] = n;
        }
    }

    // Loop through set of node IDs with no incoming edges,
    // then iterate through their children. For each child decrement their
    // incoming edge count. If incoming edge count of child == 0,
    // add its node ID to the sorted order list.

    for (int32_t n = 0; n < sorted_poa_position; n++)
    {
        int32_t node = sorted_poa[n];
        for (int32_t edge = 0; edge < outgoing_edge_count[node]; edge++)
        {
            int32_t out_node = outgoing_edges[node * CUDAPOA_MAX_NODE_EDGES + edge];
            //printf("%d\n", out_node);
            uint16_t in_node_count = local_incoming_edge_count[out_node];
            if (--in_node_count == 0)
            {
                sorted_poa_node_map[out_node]     = sorted_poa_position;
                sorted_poa[sorted_poa_position++] = out_node;
            }
            local_incoming_edge_count[out_node] = in_node_count;
        }
    }

    // sorted_poa will have final ordering of node IDs.
}

// Implementation of topological sort that matches the original
// racon source topological sort. This is helpful in ensuring the
// correctness of the GPU implementation. With this change,
// the GPU code exactly matches the SISD implementation of spoa.
template <typename SizeT>
__device__ void raconTopologicalSortDeviceUtil(SizeT* sorted_poa,
                                               SizeT* sorted_poa_node_map,
                                               SizeT node_count,
                                               uint16_t* incoming_edge_count,
                                               SizeT* incoming_edges,
                                               uint16_t* aligned_node_count,
                                               SizeT* aligned_nodes,
                                               uint8_t* node_marks,
                                               bool* check_aligned_nodes,
                                               SizeT* nodes_to_visit,
                                               SizeT max_nodes_per_graph)
{
    SizeT node_idx       = -1;
    SizeT sorted_poa_idx = 0;

    for (SizeT i = 0; i < max_nodes_per_graph; i++)
    {
        node_marks[i]          = 0;
        check_aligned_nodes[i] = true;
    }

    for (SizeT i = 0; i < node_count; i++)
    {
        if (node_marks[i] != 0)
        {
            continue;
        }

        node_idx++;
        nodes_to_visit[node_idx] = i;

        while (node_idx != -1)
        {
            SizeT node_id = nodes_to_visit[node_idx];
            bool valid    = true;

            if (node_marks[node_id] != 2)
            {
                for (int32_t e = 0; e < incoming_edge_count[node_id]; e++)
                {
                    SizeT begin_node_id = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + e];
                    if (node_marks[begin_node_id] != 2)
                    {
                        node_idx++;
                        nodes_to_visit[node_idx] = begin_node_id;
                        valid                    = false;
                    }
                }

                if (check_aligned_nodes[node_id])
                {
                    for (int32_t a = 0; a < aligned_node_count[node_id]; a++)
                    {
                        SizeT aid = aligned_nodes[node_id * CUDAPOA_MAX_NODE_ALIGNMENTS + a];
                        if (node_marks[aid] != 2)
                        {
                            node_idx++;
                            nodes_to_visit[node_idx] = aid;
                            check_aligned_nodes[aid] = false;
                            valid                    = false;
                        }
                    }
                }

                if (valid)
                {
                    node_marks[node_id] = 2;
                    if (check_aligned_nodes[node_id])
                    {
                        sorted_poa[sorted_poa_idx]   = node_id;
                        sorted_poa_node_map[node_id] = sorted_poa_idx;
                        sorted_poa_idx++;
                        for (int32_t a = 0; a < aligned_node_count[node_id]; a++)
                        {
                            SizeT aid                  = aligned_nodes[node_id * CUDAPOA_MAX_NODE_ALIGNMENTS + a];
                            sorted_poa[sorted_poa_idx] = aid;
                            sorted_poa_node_map[aid]   = sorted_poa_idx;
                            sorted_poa_idx++;
                        }
                    }
                }
                else
                {
                    node_marks[node_id] = 1;
                }
            }

            if (valid)
            {
                node_idx--;
            }
        }
    }
}

template <typename SizeT>
__global__ void runTopSortKernel(SizeT* sorted_poa,
                                 SizeT* sorted_poa_node_map,
                                 SizeT node_count,
                                 uint16_t* incoming_edge_count,
                                 SizeT* outgoing_edges,
                                 uint16_t* outgoing_edge_count,
                                 uint16_t* local_incoming_edge_count)
{
    //calls the topsort device function
    topologicalSortDeviceUtil(sorted_poa,
                              sorted_poa_node_map,
                              node_count,
                              incoming_edge_count,
                              outgoing_edges,
                              outgoing_edge_count,
                              local_incoming_edge_count);
}

// host function that calls runTopSortKernel
template <typename SizeT>
void runTopSort(SizeT* sorted_poa,
                SizeT* sorted_poa_node_map,
                SizeT node_count,
                uint16_t* incoming_edge_count,
                SizeT* outgoing_edges,
                uint16_t* outgoing_edge_count,
                uint16_t* local_incoming_edge_count)
{
    // calls the topsort kernel on 1 thread
    runTopSortKernel<<<1, 1>>>(sorted_poa,
                               sorted_poa_node_map,
                               node_count,
                               incoming_edge_count,
                               outgoing_edges,
                               outgoing_edge_count,
                               local_incoming_edge_count);
    GW_CU_CHECK_ERR(cudaPeekAtLastError());
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
