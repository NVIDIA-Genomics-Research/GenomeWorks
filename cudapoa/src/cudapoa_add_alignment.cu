
#include "cudapoa_kernels.cuh"
#include <stdio.h>

namespace nvidia {

namespace cudapoa {

/**
 * @brief Device function for adding a new alignment to the partial order alignment graph.
 *
 * @param[in/out] nodes                       Device buffer with unique nodes in graph
 * @param[in] node_count                  Number of nodes in graph
 * @graph[in] node_alignments             Device buffer with alignment nodes per node in graph
 * @param[in] node_alignment_count        Device buffer with number of aligned nodes
 * @param[in] incoming_edges              Device buffer with incoming edges per node
 * @param[in] incoming_edges_count        Device buffer with number of incoming edges per node
 * @param[in] outgoing_edges              Device buffer with outgoing edges per node
 * @param[in] outgoing_edges_count        Device buffer with number of outgoing edges per node
 * @param[in] incoming_edge_w             Device buffer with weight of incoming edges
 * @param[in] outgoing_edge_w             Device buffer with weight of outgoing edges
 * @param[in] alignment_length            Total length of new alignment
 * @param[in] graph                       Device scratch space with sorted graph
 * @param[in] alignment_graph             Device buffer with nodes from graph in alignment
 * @param[in] read                        Device scratch space with sequence
 * @param[in] alignment_read              Device buffer with bases from read in alignment
 * @param[in] node_coverage_count         Device buffer with coverage of each node in graph
 *
 * @return Number of nodes in graph after update
 */
__device__
uint16_t addAlignmentToGraph(uint8_t* nodes,
                         uint16_t node_count,
                         uint16_t* node_alignments, uint16_t* node_alignment_count,
                         uint16_t* incoming_edges, uint16_t* incoming_edge_count,
                         uint16_t* outgoing_edges, uint16_t* outgoing_edge_count,
                         uint16_t* incoming_edge_w, uint16_t* outgoing_edge_w,
                         uint16_t alignment_length,
                         uint16_t* graph,
                         int16_t* alignment_graph,
                         uint8_t* read,
                         int16_t* alignment_read,
                         uint16_t* node_coverage_counts)
{
    //printf("Running addition for alignment %d\n", alignment_length);
    int16_t head_node_id = -1;
    int16_t curr_node_id = -1;
    uint16_t prev_weight = 0;

#pragma message("TODO: Send node weights into kernel as vector. Currently hard coded.")
    const uint16_t NODE_WEIGHT = 1;

    // Basic algorithm is to iterate through the alignment of the read.
    // For each position in that alignment -
    //     if it's an insert in the read
    //         add a new node
    //     if it is aligned
    //         check if node base matches read base. if so, move on.
    //         if node base doesn't match, check other aligned nodes
    //             if none of the other aligned nodes match, add new node
    //             else use one of aligned nodes and move on.
    for(int16_t pos = alignment_length - 1; pos >= 0; pos--)
    {
        bool new_node = false;
        int16_t read_pos = alignment_read[pos];
        // Case where base in read in an insert.
        if (read_pos != -1)
        {
            //printf("%c ", read[read_pos]);
            uint8_t read_base = read[read_pos];
            int16_t graph_node_id = alignment_graph[pos];
            if (graph_node_id == -1)
            {
                // No alignment node found in graph.
                // Create new node.
                curr_node_id = node_count++;
                //printf("create new node %d\n", curr_node_id);
                new_node = true;
                nodes[curr_node_id] = read_base;
                outgoing_edge_count[curr_node_id] = 0;
                incoming_edge_count[curr_node_id] = 0;
                node_alignment_count[curr_node_id] = 0;
                node_coverage_counts[curr_node_id] = 0;
            }
            else
            {
                // Get base information for aligned node in graph.
                uint8_t graph_base = nodes[graph_node_id];
                //printf("graph base %c\n", graph_base);

                // If bases match, then set current node id to graph node id.
                if (graph_base == read_base)
                {
                    //printf("graph and read base are same\n");
                    curr_node_id = graph_node_id;
                }
                else
                {
                    // Since bases don't match, iterate through all aligned nodes of
                    // graph node, and check against their bases. If a base matches,
                    // then set the current node as that aligned node.
                    uint16_t num_aligned_node = node_alignment_count[graph_node_id];
                    //printf("aligned nodes are %d\n", num_aligned_node);
                    int16_t aligned_node_id = -1;
                    //printf("looping through alignments\n");
                    for(uint16_t n = 0; n < num_aligned_node; n++)
                    {
                        uint16_t aid = node_alignments[graph_node_id * CUDAPOA_MAX_NODE_ALIGNMENTS + n];
                        if (nodes[aid] == read_base)
                        {
                            aligned_node_id = aid;
                            break;
                        }
                    }

                    if (aligned_node_id != -1)
                    {
                        //printf("found aligned node %d\n", aligned_node_id);
                        curr_node_id = aligned_node_id;
                    }
                    else
                    {
                        // However, if none of the nodes in the aligned list match either,
                        // then create a new node and update the graph node (+ aligned nodes)
                        // with information about this new node since it also becomes an aligned
                        // node to the others.
                        new_node = true;
                        curr_node_id = node_count++;
                        //printf("create new node %d\n", curr_node_id);
                        nodes[curr_node_id] = read_base;
                        outgoing_edge_count[curr_node_id] = 0;
                        incoming_edge_count[curr_node_id] = 0;
                        node_alignment_count[curr_node_id] = 0;
                        node_coverage_counts[curr_node_id] = 0;
                        uint16_t new_node_alignments = 0;

                        for(uint16_t n = 0; n < num_aligned_node; n++)
                        {
                            uint16_t aid = node_alignments[graph_node_id * CUDAPOA_MAX_NODE_ALIGNMENTS + n];
                            uint16_t aid_count = node_alignment_count[aid];
                            node_alignments[aid * CUDAPOA_MAX_NODE_ALIGNMENTS + aid_count] = curr_node_id;
                            node_alignment_count[aid] = aid_count + 1;
                            node_alignments[curr_node_id * CUDAPOA_MAX_NODE_ALIGNMENTS + new_node_alignments] = aid;
                            new_node_alignments++;
                        }

                        node_alignments[graph_node_id * CUDAPOA_MAX_NODE_ALIGNMENTS + num_aligned_node] = curr_node_id;
                        node_alignment_count[graph_node_id] = num_aligned_node + 1;

                        node_alignments[curr_node_id * CUDAPOA_MAX_NODE_ALIGNMENTS + new_node_alignments] = graph_node_id;
                        new_node_alignments++;

                        node_alignment_count[curr_node_id] = new_node_alignments;
                    }
                }
            }

            if (new_node)
            {
                //printf("new node %d\n", curr_node_id);
            }

            // Create new edges if necessary.
            if (head_node_id != -1)
            {
                bool edge_exists = false;
                uint16_t in_count = incoming_edge_count[curr_node_id];
                for(uint16_t e = 0; e < in_count; e++)
                {
                    if(incoming_edges[curr_node_id * CUDAPOA_MAX_NODE_EDGES + e] == head_node_id)
                    {
                        edge_exists = true;
                        incoming_edge_w[curr_node_id * CUDAPOA_MAX_NODE_EDGES + e] += (prev_weight + NODE_WEIGHT);
                        //printf("Update existing node from %d to %d with weight %d\n", head_node_id, curr_node_id, incoming_edge_w[curr_node_id * CUDAPOA_MAX_NODE_EDGES + e]);
                    }
                }
                if (!edge_exists)
                {
                    incoming_edges[curr_node_id * CUDAPOA_MAX_NODE_EDGES + in_count] = head_node_id;
                    incoming_edge_w[curr_node_id * CUDAPOA_MAX_NODE_EDGES + in_count] = prev_weight + NODE_WEIGHT;
                    incoming_edge_count[curr_node_id] = in_count + 1;
                    uint16_t out_count = outgoing_edge_count[head_node_id];
                    outgoing_edges[head_node_id * CUDAPOA_MAX_NODE_EDGES + out_count] = curr_node_id;
                    outgoing_edge_count[head_node_id] = out_count + 1;
                    //printf("Created new edge %d to %d with weight %d\n", head_node_id, curr_node_id, prev_weight + NODE_WEIGHT);

                    if (out_count + 1 >= CUDAPOA_MAX_NODE_EDGES || in_count + 1 >= CUDAPOA_MAX_NODE_EDGES)
                    {
                        printf("exceeded max edge count\n");
                    }
                }

            }

            head_node_id = curr_node_id;

            // If a node is seen within a graph, then it's part of some
            // read, hence its coverage is incremented by 1.
            node_coverage_counts[head_node_id]++;
        }

        prev_weight = NODE_WEIGHT;

    }
    //printf("final size %d\n", node_count);
    return node_count;
}

}

}
