// Implementation file for CUDA POA kernels.

#include "cudapoa_kernels.cuh"
#include "cudapoa_nw.cu"
#include "cudapoa_topsort.cu"
#include "cudapoa_add_alignment.cu"
#include "cudapoa_generate_consensus.cu"

#include <stdio.h>

namespace nvidia {

namespace cudapoa {

/**
 * @brief The main kernel that runs the partial order alignment
 *        algorithm.
 *
 * @param[out] consensus_d                Device buffer for generated consensus
 * @param[out] coverage_d                 Device buffer for coverage of each base in consensus
 * @param[in] sequences_d                 Device buffer with sequences for all windows
 * @param[in] sequence_lengths_d          Device buffer sequence lengths
 * @param[in] window_details_d            Device buffer with structs 
 *                                        encapsulating sequence details per window
 * @param[in] total_window                Total number of windows to process
 * @param[in] scores                      Device scratch space that scores alignment matrix score
 * @param[in] alignment_graph_d           Device scratch space for backtrace alignment of graph
 * @param[in] alignment_read_d            Device scratch space for backtrace alignment of sequence
 * @param[in] nodes                       Device scratch space for storing unique nodes in graph
 * @param[in] incoming_edges              Device scratch space for storing incoming edges per node
 * @param[in] incoming_edges_count        Device scratch space for storing number of incoming edges per node
 * @param[in] outgoing_edges              Device scratch space for storing outgoing edges per node
 * @param[in] outgoing_edges_count        Device scratch space for storing number of outgoing edges per node
 * @param[in] incoming_edge_w             Device scratch space for storing weight of incoming edges
 * @param[in] outgoing_edge_w             Device scratch space for storing weight of outgoing edges
 * @param[in] sorted_poa                  Device scratch space for storing sorted graph
 * @param[in] node_id_to_pos              Device scratch space for mapping node ID to position in graph
 * @graph[in] node_alignments             Device scratch space for storing alignment nodes per node in graph
 * @param[in] node_alignment_count        Device scratch space for storing number of aligned nodes
 * @param[in] sorted_poa_local_edge_count Device scratch space for maintaining edge counts during topological sort
 * @param[in] consensus_scores            Device scratch space for storing score of each node while traversing graph during consensus
 * @param[in] consensus_predecessors      Device scratch space for storing predecessors of nodes while traversing graph during consensus
 * @param[in] node_marks_d_               Device scratch space for storing node marks when running spoa accurate top sort
 * @param[in] check_aligned_nodes_d_      Device scratch space for storing check for aligned nodes
 * @param[in] nodes_to_visit_d_           device scratch space for storing stack of nodes to be visited in topsort
 * @param[in] node_coverage_counts_d_     device scratch space for storing coverage of each node in graph.
 */
__global__
void generatePOAKernel(uint8_t* consensus_d,
                       uint16_t* coverage_d,
                       uint8_t* sequences_d,
                       uint16_t * sequence_lengths_d,
                       nvidia::cudapoa::WindowDetails * window_details_d,
                       uint32_t total_windows,
                       int16_t* scores_d,
                       int16_t* alignment_graph_d,
                       int16_t* alignment_read_d,
                       uint8_t* nodes_d,
                       uint16_t* incoming_edges_d,
                       uint16_t* incoming_edge_count_d,
                       uint16_t* outgoing_edges_d,
                       uint16_t* outgoing_edge_count_d,
                       uint16_t* incoming_edge_w_d,
                       uint16_t* outgoing_edge_w_d,
                       uint16_t* sorted_poa_d,
                       uint16_t* node_id_to_pos_d,
                       uint16_t* node_alignments_d,
                       uint16_t* node_alignment_count_d,
                       uint16_t* sorted_poa_local_edge_count_d,
                       int32_t* consensus_scores_d,
                       int16_t* consensus_predecessors_d,
                       uint8_t* node_marks_d_,
                       bool* check_aligned_nodes_d_,
                       uint16_t* nodes_to_visit_d_,
                       uint16_t* node_coverage_counts_d_)
{

    uint32_t block_idx = blockIdx.x;
    uint32_t thread_idx = threadIdx.x;

    long long int back_time = 0;
    long long int nw_time = 0;
    long long int add_time = 0;
    long long int top_time = 0;

    if (block_idx > total_windows)
        return;

    // Find the buffer offsets for each thread within the global memory buffers.
    uint8_t* nodes = &nodes_d[CUDAPOA_MAX_NODES_PER_WINDOW * block_idx];
    uint16_t* incoming_edges = &incoming_edges_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* incoming_edge_count = &incoming_edge_count_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW];
    uint16_t* outoing_edges = &outgoing_edges_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_count = &outgoing_edge_count_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW];
    uint16_t* incoming_edge_weights = &incoming_edge_w_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_weights = &outgoing_edge_w_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* sorted_poa = &sorted_poa_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW];
    uint16_t* node_id_to_pos = &node_id_to_pos_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW];
    uint16_t* node_alignments = &node_alignments_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW * CUDAPOA_MAX_NODE_ALIGNMENTS];
    uint16_t* node_alignment_count = &node_alignment_count_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW];
    uint16_t* sorted_poa_local_edge_count = &sorted_poa_local_edge_count_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW];

    int16_t* scores = &scores_d[CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION * block_idx];
    int16_t* alignment_graph = &alignment_graph_d[CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION * block_idx];
    int16_t* alignment_read = &alignment_read_d[CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION * block_idx];
    uint16_t* node_coverage_counts = &node_coverage_counts_d_[CUDAPOA_MAX_NODES_PER_WINDOW * block_idx];

    uint8_t* node_marks = &node_marks_d_[CUDAPOA_MAX_NODES_PER_WINDOW * block_idx];
    bool* check_aligned_nodes = &check_aligned_nodes_d_[CUDAPOA_MAX_NODES_PER_WINDOW * block_idx];
    uint16_t* nodes_to_visit = &nodes_to_visit_d_[CUDAPOA_MAX_NODES_PER_WINDOW * block_idx];

    //get Block-specific variables
    uint32_t window_idx = blockIdx.x;

    uint16_t * sequence_lengths = &sequence_lengths_d[window_details_d[window_idx].seq_len_buffer_offset];

    uint32_t num_sequences = window_details_d[window_idx].num_seqs;
    uint8_t * sequence = &sequences_d[window_details_d[window_idx].seq_starts];

    long long int t0 = clock64();

    if (thread_idx == 0)
    {

        // Create backbone for window based on first sequence in window.
        nodes[0] = sequence[0];
        sorted_poa[0] = 0;
        incoming_edge_count[0] = 0;
        node_alignment_count[0] = 0;
        node_id_to_pos[0] = 0;
        outgoing_edge_count[sequence_lengths[0] - 1] = 0;
        incoming_edge_weights[0] = 0;
        node_coverage_counts[0] = 1;

        //Build the rest of the graphs
        for (uint16_t nucleotide_idx=1; nucleotide_idx<sequence_lengths[0]; nucleotide_idx++){
            nodes[nucleotide_idx] = sequence[nucleotide_idx];
            sorted_poa[nucleotide_idx] = nucleotide_idx;
            outoing_edges[(nucleotide_idx-1) * CUDAPOA_MAX_NODE_EDGES] = nucleotide_idx;
            outgoing_edge_count[nucleotide_idx-1] = 1;
            incoming_edges[nucleotide_idx * CUDAPOA_MAX_NODE_EDGES] = nucleotide_idx - uint16_t(1);
            incoming_edge_weights[nucleotide_idx * CUDAPOA_MAX_NODE_EDGES] = 0;
            incoming_edge_count[nucleotide_idx] = 1;
            node_alignment_count[nucleotide_idx] = 0;
            node_id_to_pos[nucleotide_idx] = nucleotide_idx;
            node_coverage_counts[nucleotide_idx] = 1;
        }

    }

    __syncthreads();

    back_time += (clock64() - t0);

    //for(uint16_t i = 0; i < sequence_lengths[0]; i++)
    //{
    //    printf("%c ", nodes[i]);
    //}

    // Generate consensus only if sequences are aligned to graph.
    bool generate_consensus = false;

    //printf("window id %d, sequence %d\n", block_idx, num_sequences_in_window - 1);

    // Align each subsequent read, add alignment to graph, run topoligical sort.
    for(uint16_t s = 1; s < num_sequences; s++)
    {
        //printf("running window %d seq %d / %d\n", block_idx, s, num_sequences_in_window);
        uint16_t seq_len = sequence_lengths[s];
        sequence += sequence_lengths[s - 1]; // increment the pointer so it is pointing to correct sequence data
/*
        if (thread_idx == 0)
            printf("seq len is %i for sequence %i\n", seq_len, s);
*/

        //for(uint16_t i = 0; i < seq_len; i++)
        //{
        //    printf("%c ", seq[i]);
        //}

        //return;
        // Run DP step and fetch traceback.
        //bool found_node = false;
        //for(uint16_t i = 0; i < sequence_length_data[0]; i++)
        //{
        //    if (outgoing_edge_count[i] == 0)
        //    {
        //        printf("node %d has 0 oe\n", i);
        //        found_node = true;
        //    }
        //}
        //if (!found_node)
        //{
        //    printf("DID NOT FIND A NODE WITH NO OUTGOING EDGE before alignment!!!!\n");
        //    return;
        //}

        // print sorted graph
        //for(uint16_t i = 0; i < sequence_length_data[0]; i++)
        //{
        //    printf("%d ", sorted_poa[i]);
        //}
        //printf("\n");

        if (thread_idx == 0)
        {

            if (sequence_lengths[0] >= CUDAPOA_MAX_NODES_PER_WINDOW)
            {
                printf("Node count %d is greater than max matrix size %d\n", sequence_lengths[0], CUDAPOA_MAX_NODES_PER_WINDOW);
                return;
            }
            if (seq_len >= CUDAPOA_MAX_NODES_PER_WINDOW)
            {
                printf("Sequence len %d is greater than max matrix size %d\n", seq_len, CUDAPOA_MAX_NODES_PER_WINDOW);
                return;
            }


        }
        long long int start = clock64();

        // Run Needleman-Wunsch alignment between graph and new sequence.
/*
        if (thread_idx ==0)
            printf("running nw with sequence length of %i and sequence of %c %c %c %c %c\n", seq_len, sequence[0], sequence[1], sequence[2], sequence[3], sequence[4]);
*/

        uint16_t alignment_length = runNeedlemanWunsch(nodes,
                sorted_poa,
                node_id_to_pos,
                sequence_lengths[0],
                incoming_edge_count,
                incoming_edges,
                outgoing_edge_count,
                outoing_edges,
                sequence,
                seq_len,
                scores,
                alignment_graph,
                alignment_read);

        long long int nw_end = clock64();
        nw_time += (nw_end - start);

        __syncthreads();

        //found_node = false;
        //for(uint16_t i = 0; i < sequence_length_data[0]; i++)
        //{
        //    if (outgoing_edge_count[i] == 0)
        //    {
        //        printf("node %d has 0 oe\n", i);
        //        found_node = true;
        //    }
        //}
        //if (!found_node)
        //{
        //    printf("DID NOT FIND A NODE WITH NO OUTGOING EDGE before addition!!!!\n");
        //    return;
        //}

        start = clock64();

        if (thread_idx == 0)
        {

            // Add alignment to graph.
            //printf("running add\n");
            sequence_lengths[0] = addAlignmentToGraph(nodes, sequence_lengths[0],
                    node_alignments, node_alignment_count,
                    incoming_edges, incoming_edge_count,
                    outoing_edges, outgoing_edge_count,
                    incoming_edge_weights, outgoing_edge_weights,
                    alignment_length,
                    sorted_poa, alignment_graph, 
                    sequence, alignment_read,
                    node_coverage_counts);

            long long int add_end = clock64();
            add_time += (add_end - start);

            // Verify that each graph has at least one node with no outgoing edges.
            //bool found_node = false;
            //for(uint16_t i = 0; i < sequence_length_data[0]; i++)
            //{
            //    //printf("node id %d ie %d oe %d\n ", i, incoming_edge_count[i], outgoing_edge_count[i]);
            //    if (outgoing_edge_count[i] == 0)
            //        found_node = true;
            //}
            //if (!found_node)
            //{
            //    printf("DID NOT FIND A NODE WITH NO OUTGOING EDGE after addition!!!!\n");
            //    return;
            //}


            // Run a topsort on the graph. Not strictly necessary at this point
            //printf("running topsort\n");
#ifdef SPOA_ACCURATE
            // Exactly matches racon CPU results
            raconTopologicalSortDeviceUtil(sorted_poa,
                                      node_id_to_pos,
                                      sequence_lengths[0],
                                      incoming_edge_count,
                                      incoming_edges,
                                      node_alignment_count,
                                      node_alignments,
                                      node_marks,
                                      check_aligned_nodes,
                                      nodes_to_visit);
#else
            // Faster top sort
            topologicalSortDeviceUtil(sorted_poa,
                                      node_id_to_pos,
                                      sequence_lengths[0],
                                      incoming_edge_count,
                                      outoing_edges,
                                      outgoing_edge_count,
                                      sorted_poa_local_edge_count);
#endif

            long long int top_end = clock64();
            top_time += (top_end - add_end);
            //printf("done loop\n");
        }

        __syncthreads();

        generate_consensus = true;
    }

    // Dummy kernel code to copy first sequence as output.
    //uint8_t *input_row = &sequences_d[input_row_idx * sequences_pitch];
    //uint8_t *output_row = &consensus_d[block_idx * consensus_pitch];
    //for(uint32_t c = 0; c < sequence_lengths_d[block_idx * max_depth_per_window]; c++)
    //{
    //    output_row[c] = input_row[c];
    //}

    long long int consensus_time = 0;

    if (thread_idx == 0 && generate_consensus)
    {
        uint8_t* consensus = &consensus_d[block_idx * CUDAPOA_MAX_SEQUENCE_SIZE];
        uint16_t* coverage = &coverage_d[block_idx * CUDAPOA_MAX_SEQUENCE_SIZE];
        int32_t* consensus_scores = &consensus_scores_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW];
        int16_t* consensus_predecessors = &consensus_predecessors_d[block_idx * CUDAPOA_MAX_NODES_PER_WINDOW];

        long long int start = clock64();
        generateConsensus(nodes,
                sequence_lengths[0],
                sorted_poa,
                node_id_to_pos,
                incoming_edges,
                incoming_edge_count,
                outoing_edges,
                outgoing_edge_count,
                incoming_edge_weights,
                consensus_predecessors,
                consensus_scores,
                consensus,
                coverage,
                node_coverage_counts,
                node_alignments, node_alignment_count);
        consensus_time = (clock64() - start);
    }
    //if (thread_idx == 0)
    //{
    //    long long int total = back_time + nw_time + add_time + top_time + consensus_time;
    //    printf("Total time of backbone generation is %lf %\n", ((double)back_time / total) * 100.f);
    //    printf("Total time of nw is %lf %\n", ((double)nw_time / total) * 100.f);
    //    printf("Total time of addition is %lf %\n", ((double)add_time / total) * 100.f);
    //    printf("Total time of topsort is %lf %\n", ((double)top_time / total) * 100.f);
    //    printf("Total time of consensus is %lf %\n", ((double)consensus_time / total) * 100.f);
    //}

}

// Host function call for POA kernel.
void generatePOA(uint8_t* consensus_d,
                 uint16_t* coverage_d,
                 uint8_t* sequences_d,
                 uint16_t * sequence_lengths_d,
                 nvidia::cudapoa::WindowDetails * window_details_d,
                 uint32_t total_windows,
                 uint32_t num_threads,
                 uint32_t num_blocks,
                 cudaStream_t stream,
                 int16_t* scores,
                 int16_t* alignment_graph,
                 int16_t* alignment_read,
                 uint8_t* nodes,
                 uint16_t* incoming_edges,
                 uint16_t* incoming_edge_count,
                 uint16_t* outgoing_edges,
                 uint16_t* outgoing_edge_count,
                 uint16_t* incoming_edge_w,
                 uint16_t* outgoing_edge_w,
                 uint16_t* sorted_poa,
                 uint16_t* node_id_to_pos,
                 uint16_t* node_alignments,
                 uint16_t* node_alignment_count,
                 uint16_t* sorted_poa_local_edge_count,
                 int32_t* consensus_scores,
                 int16_t* consensus_predecessors,
                 uint8_t* node_marks,
                 bool* check_aligned_nodes,
                 uint16_t* nodes_to_visit,
                 uint16_t* node_coverage_counts)
{
    generatePOAKernel<<<num_blocks, num_threads, 0, stream>>>(consensus_d,
                                                              coverage_d,
                                                              sequences_d,
                                                              sequence_lengths_d,
                                                              window_details_d,
                                                              total_windows,
                                                              scores,
                                                              alignment_graph,
                                                              alignment_read,
                                                              nodes,
                                                              incoming_edges,
                                                              incoming_edge_count,
                                                              outgoing_edges,
                                                              outgoing_edge_count,
                                                              incoming_edge_w,
                                                              outgoing_edge_w,
                                                              sorted_poa,
                                                              node_id_to_pos,
                                                              node_alignments,
                                                              node_alignment_count,
                                                              sorted_poa_local_edge_count,
                                                              consensus_scores,
                                                              consensus_predecessors,
                                                              node_marks,
                                                              check_aligned_nodes,
                                                              nodes_to_visit,
                                                              node_coverage_counts);
}

} // namespace cudapoa

} // namespace nvidia
