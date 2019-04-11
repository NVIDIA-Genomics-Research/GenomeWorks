
#include "cudapoa_kernels.cuh"
#include <stdio.h>

namespace nvidia {

namespace cudapoa {

__device__
uint16_t branchCompletion(uint16_t max_score_id_pos,
                         uint8_t* nodes,
                         uint16_t node_count,
                         uint16_t* graph,
                         uint16_t* incoming_edges,
                         uint16_t* incoming_edge_count,
                         uint16_t* outgoing_edges,
                         uint16_t* outgoing_edge_count,
                         uint16_t* incoming_edge_w,
                         int32_t* scores,
                         int16_t* predecessors)
{
    uint16_t node_id = graph[max_score_id_pos];

    // Go through all the outgoing edges of the node, and for
    // each of the end nodes of the edges clear the scores
    // for all the _other_ nodes that had edges to that end node.
    uint16_t out_edges = outgoing_edge_count[node_id];
    for(uint16_t oe = 0; oe < out_edges; oe++)
    {
        uint16_t out_node_id = outgoing_edges[node_id * CUDAPOA_MAX_NODE_EDGES + oe];
        uint16_t out_node_in_edges = incoming_edge_count[out_node_id];
        for(uint16_t ie = 0; ie < out_node_in_edges; ie++)
        {
            uint16_t id = incoming_edges[out_node_id * CUDAPOA_MAX_NODE_EDGES + ie];
            if (id != node_id)
            {
                scores[id] = -1;
            }
        }
    }

    int32_t max_score = 0;
    uint16_t max_score_id = 0;
    // Run the same node weight traversal algorithm as always, to find the new
    // node with maximum weight.
    // We can start from the very next position in the graph rank because
    // the graph is topologically sorted and hence guarantees that successor of the current max
    // node will be processed again.
    for(uint16_t graph_pos = max_score_id_pos + 1; graph_pos < node_count; graph_pos++)
    {
        node_id = graph[graph_pos];

        predecessors[node_id] = -1;
        int32_t score_node_id = -1;

        uint16_t in_edges = incoming_edge_count[node_id];
        for(uint16_t e = 0; e < in_edges; e++)
        {
            uint16_t begin_node_id = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + e];
            if (scores[begin_node_id] == -1)
            {
                continue;
            }

            int32_t edge_w = static_cast<int32_t>(incoming_edge_w[node_id * CUDAPOA_MAX_NODE_EDGES + e]);
            if (score_node_id < edge_w ||
                    (score_node_id == edge_w &&
                     scores[predecessors[node_id]] <= scores[begin_node_id]))
            {
                score_node_id = edge_w;
                predecessors[node_id] = begin_node_id;
            }
        }

        if (predecessors[node_id] != -1)
        {
            score_node_id += scores[predecessors[node_id]];
        }

        if (max_score < score_node_id)
        {
            max_score = score_node_id;
            max_score_id = node_id;
        }
        //printf("max score %d, max score id %d, node id %d score %d\n", max_score, max_score_id, node_id, score_node_id);

        scores[node_id] = score_node_id;
    }

    return max_score_id;
}


/**
 * @brief Device function to generate consensus from a given graph.
 *        The input graph needs to be topologically sorted.
 *
 * @param[in] nodes                 Device buffer with unique nodes in graph
 * @param[in] node_count            Number of nodes in graph
 * @param[in] graph                 Device buffer with sorted graph
 * @param[in] node_id_to_pos        Device scratch space for mapping node ID to position in graph
 * @param[in] incoming_edges        Device buffer with incoming edges per node
 * @param[in] incoming_edges_count  Device buffer with number of incoming edges per node
 * @param[in] outgoing_edges        Device buffer with outgoing edges per node
 * @param[in] outgoing_edges_count  Device buffer with number of outgoing edges per node
 * @param[in] predecessors          Device buffer with predecessors of nodes while traversing graph during consensus
 * @param[in] scores                Device buffer with score of each node while traversing graph during consensus
 * @param[out] consensus            Device buffer for generated consensus
 * @param[out] coverate             Device buffer for coverage of each base in consensus
 * @param[out] node_coverage_counts Device buffer with coverage of each base in graph
 * @param[in] node_alignments       Device buffer with aligned nodes for each node in graph
 * @param[in] node_alignment)count  Device buffer with aligned nodes count for each node in graph
 */
__device__
void generateConsensus(uint8_t* nodes,
                         uint16_t node_count,
                         uint16_t* graph,
                         uint16_t* node_id_to_pos,
                         uint16_t* incoming_edges,
                         uint16_t* incoming_edge_count,
                         uint16_t* outgoing_edges,
                         uint16_t* outgoing_edge_count,
                         uint16_t* incoming_edge_w,
                         int16_t* predecessors,
                         int32_t* scores,
                         uint8_t* consensus,
                         uint16_t* coverage,
                         uint16_t* node_coverage_counts,
                         uint16_t* node_alignments,
                         uint16_t* node_alignment_count)
{
    // Initialize scores and predecessors to default value.
    for(uint16_t i = 0; i < node_count; i++)
    {
        predecessors[i] = -1;
        scores[i] = -1;
    }

    uint16_t max_score_id = 0;
    int32_t max_score = -1;

    for(uint16_t graph_pos = 0; graph_pos < node_count; graph_pos++)
    {
        uint16_t node_id = graph[graph_pos];
        uint16_t in_edges = incoming_edge_count[node_id];

        int32_t score_node_id = scores[node_id];

        // For each node, go through it's incoming edges.
        // If the weight of any of the incoming edges is greater
        // than the score of the current node, or if the weight is equal
        // but the predecessors of the edge are heavier than the current node,
        // then update the score of the node to be the incoming edge weight.
        for(uint16_t e = 0; e < in_edges; e++)
        {
            int32_t edge_w = static_cast<int32_t>(incoming_edge_w[node_id * CUDAPOA_MAX_NODE_EDGES + e]);
            uint16_t begin_node_id = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + e];
            if (score_node_id < edge_w ||
                    (score_node_id == edge_w &&
                     scores[predecessors[node_id]] <= scores[begin_node_id]))
            {
                score_node_id = edge_w;
                predecessors[node_id] = begin_node_id;
            }
        }

        // Then update the score of the node to be the sum
        // of the score of the predecessor and itself.
        if (predecessors[node_id] != -1)
        {
            score_node_id += scores[predecessors[node_id]];
        }

        // Keep track of the highest weighted node.
        if (max_score < score_node_id)
        {
            max_score_id = node_id;
            max_score = score_node_id;
        }
        //printf("max score %d, max score id %d, node id %d score %d\n", max_score, max_score_id, node_id, score_node_id);

        scores[node_id] = score_node_id;
    }

    // If the node with maximum score isn't a leaf of the graph
    // then run a special branch completion function.
    if (outgoing_edge_count[max_score_id] != 0)
    {
        while(outgoing_edge_count[max_score_id] != 0)
        {
            max_score_id = branchCompletion(node_id_to_pos[max_score_id],
                                             nodes,
                                             node_count,
                                             graph,
                                             incoming_edges,
                                             incoming_edge_count,
                                             outgoing_edges,
                                             outgoing_edge_count,
                                             incoming_edge_w,
                                             scores,
                                             predecessors);
        }
    }

    uint16_t consensus_pos = 0;
    while(predecessors[max_score_id] != -1)
    {
        consensus[consensus_pos] = nodes[max_score_id];
        uint16_t cov = node_coverage_counts[max_score_id];
        for(uint16_t a = 0; a < node_alignment_count[max_score_id]; a++)
        {
            cov += node_coverage_counts[node_alignments[max_score_id * CUDAPOA_MAX_NODE_ALIGNMENTS + a]];
        }
        coverage[consensus_pos] = cov;
        max_score_id = predecessors[max_score_id];
        consensus_pos++;

    }
    consensus[consensus_pos] = nodes[max_score_id];
    uint16_t cov = node_coverage_counts[max_score_id];
    for(uint16_t a = 0; a < node_alignment_count[max_score_id]; a++)
    {
        cov += node_coverage_counts[node_alignments[max_score_id * CUDAPOA_MAX_NODE_ALIGNMENTS + a]];
    }
    coverage[consensus_pos] = cov;
    consensus++;
    consensus[consensus_pos] = '\0';
}

}

}
