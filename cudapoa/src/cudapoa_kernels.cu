// Implementation file for CUDA POA kernels.

#include "cudapoa_kernels.cuh"
#include "cudapoa_nw.cu"
#include "cudapoa_nw_banded.cu"
#include "cudapoa_topsort.cu"
#include "cudapoa_add_alignment.cu"
#include "cudapoa_generate_consensus.cu"
#include <cudautils/cudautils.hpp>

#include <stdio.h>

namespace genomeworks {

namespace cudapoa {

/**
 * @brief The main kernel that runs the partial order alignment
 *        algorithm.
 *
 * @param[out] consensus_d                Device buffer for generated consensus
 * @param[out] coverage_d                 Device buffer for coverage of each base in consensus
 * @param[in] sequences_d                 Device buffer with sequences for all windows
 * @param[in] base_weights_d              Device buffer with base weights for all windows
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
 * @param[in] gap_score                   Score for inserting gap into alignment
 * @param[in] mismatch_score              Score for finding a mismatch in alignment
 * @param[in] match_score                 Score for finding a match in alignment
 */
template<int32_t TPB = 64, bool cuda_banded_alignment = false>
__global__
void generatePOAKernel(uint8_t* consensus_d,
                       uint16_t* coverage_d,
                       uint8_t* sequences_d,
                       uint8_t* base_weights_d,
                       uint16_t * sequence_lengths_d,
                       genomeworks::cudapoa::WindowDetails * window_details_d,
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
                       uint16_t* node_coverage_counts_d_,
                       int16_t gap_score,
                       int16_t mismatch_score,
                       int16_t match_score)
{

    uint32_t nwindows_per_block = TPB/WARP_SIZE;
    uint32_t warp_idx = threadIdx.x / WARP_SIZE;
    uint32_t lane_idx = threadIdx.x % WARP_SIZE;

    uint32_t window_idx = blockIdx.x * nwindows_per_block + warp_idx;


    if (window_idx >= total_windows)
        return;

    uint32_t matrix_sequence_dimension = cuda_banded_alignment ? CUDAPOA_BANDED_MAX_MATRIX_SEQUENCE_DIMENSION : CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION;
    uint32_t max_nodes_per_window = cuda_banded_alignment ? CUDAPOA_MAX_NODES_PER_WINDOW_BANDED : CUDAPOA_MAX_NODES_PER_WINDOW;
    uint32_t max_graph_dimension = cuda_banded_alignment ? CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION_BANDED : CUDAPOA_MAX_MATRIX_GRAPH_DIMENSION;

    // Find the buffer offsets for each thread within the global memory buffers.
    uint8_t* nodes = &nodes_d[max_nodes_per_window * window_idx];
    uint16_t* incoming_edges = &incoming_edges_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* incoming_edge_count = &incoming_edge_count_d[window_idx * max_nodes_per_window];
    uint16_t* outoing_edges = &outgoing_edges_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_count = &outgoing_edge_count_d[window_idx * max_nodes_per_window];
    uint16_t* incoming_edge_weights = &incoming_edge_w_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* outgoing_edge_weights = &outgoing_edge_w_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_EDGES];
    uint16_t* sorted_poa = &sorted_poa_d[window_idx * max_nodes_per_window];
    uint16_t* node_id_to_pos = &node_id_to_pos_d[window_idx * max_nodes_per_window];
    uint16_t* node_alignments = &node_alignments_d[window_idx * max_nodes_per_window * CUDAPOA_MAX_NODE_ALIGNMENTS];
    uint16_t* node_alignment_count = &node_alignment_count_d[window_idx * max_nodes_per_window];
    uint16_t* sorted_poa_local_edge_count = &sorted_poa_local_edge_count_d[window_idx * max_nodes_per_window];

    int16_t* scores = &scores_d[max_graph_dimension * matrix_sequence_dimension * window_idx];
    int16_t* alignment_graph = &alignment_graph_d[max_graph_dimension * window_idx];
    int16_t* alignment_read = &alignment_read_d[max_graph_dimension * window_idx];
    uint16_t* node_coverage_counts = &node_coverage_counts_d_[max_nodes_per_window * window_idx];

#ifdef SPOA_ACCURATE
    uint8_t* node_marks = &node_marks_d_[max_nodes_per_window * window_idx];
    bool* check_aligned_nodes = &check_aligned_nodes_d_[max_nodes_per_window * window_idx];
    uint16_t* nodes_to_visit = &nodes_to_visit_d_[max_nodes_per_window * window_idx];
#endif

    uint16_t * sequence_lengths = &sequence_lengths_d[window_details_d[window_idx].seq_len_buffer_offset];

    uint32_t num_sequences = window_details_d[window_idx].num_seqs;
    uint8_t * sequence = &sequences_d[window_details_d[window_idx].seq_starts];
    uint8_t * base_weights = &base_weights_d[window_details_d[window_idx].seq_starts];

    if (lane_idx == 0)
    {
        // Create backbone for window based on first sequence in window.
        nodes[0] = sequence[0];
        sorted_poa[0] = 0;
        incoming_edge_count[0] = 0;
        node_alignment_count[0] = 0;
        node_id_to_pos[0] = 0;
        outgoing_edge_count[sequence_lengths[0] - 1] = 0;
        incoming_edge_weights[0] = base_weights[0];
        node_coverage_counts[0] = 1;

        //Build the rest of the graphs
        for (uint16_t nucleotide_idx=1; nucleotide_idx<sequence_lengths[0]; nucleotide_idx++){
            nodes[nucleotide_idx] = sequence[nucleotide_idx];
            sorted_poa[nucleotide_idx] = nucleotide_idx;
            outoing_edges[(nucleotide_idx-1) * CUDAPOA_MAX_NODE_EDGES] = nucleotide_idx;
            outgoing_edge_count[nucleotide_idx-1] = 1;
            incoming_edges[nucleotide_idx * CUDAPOA_MAX_NODE_EDGES] = nucleotide_idx - uint16_t(1);
            incoming_edge_weights[nucleotide_idx * CUDAPOA_MAX_NODE_EDGES] = base_weights[nucleotide_idx - 1] + base_weights[nucleotide_idx];
            incoming_edge_count[nucleotide_idx] = 1;
            node_alignment_count[nucleotide_idx] = 0;
            node_id_to_pos[nucleotide_idx] = nucleotide_idx;
            node_coverage_counts[nucleotide_idx] = 1;
        }

    }

    __syncwarp();


    // Generate consensus only if sequences are aligned to graph.
    bool generate_consensus = false;

    //printf("window id %d, sequence %d\n", window_idx, num_sequences_in_window - 1);

    // Align each subsequent read, add alignment to graph, run topoligical sort.
    for(uint16_t s = 1; s < num_sequences; s++){
        uint16_t seq_len = sequence_lengths[s];
        sequence += sequence_lengths[s - 1]; // increment the pointer so it is pointing to correct sequence data
        base_weights += sequence_lengths[s - 1]; // increment the pointer so it is pointing to correct sequence data

        if (lane_idx == 0){
            if (sequence_lengths[0] >= max_nodes_per_window){
                printf("Node count %d is greater than max matrix size %d\n", sequence_lengths[0], max_nodes_per_window);
                return;
            }
            if (seq_len >= max_nodes_per_window){
                printf("Sequence len %d is greater than max matrix size %d\n", seq_len, max_nodes_per_window);
                return;
            }
        }

        // Run Needleman-Wunsch alignment between graph and new sequence.
        uint16_t alignment_length;

        if(cuda_banded_alignment){
            alignment_length = runNeedlemanWunschBanded(nodes,
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
                                                       alignment_read,
                                                       gap_score,
                                                       mismatch_score,
                                                       match_score);
        } else {
            alignment_length = runNeedlemanWunsch<uint8_t, uint16_t, int16_t, TPB>(nodes,
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
                                                alignment_read,
                                                gap_score,
                                                mismatch_score,
                                                match_score);
        }


        __syncwarp();
	//printf("%d %d %d\n", s, window_idx, alignment_length);

        if (lane_idx == 0){

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
                    node_coverage_counts,
                    base_weights);


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
                                      nodes_to_visit,
                                      cuda_banded_alignment);
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
        }

        __syncwarp();

        generate_consensus = true;
    }


    if (lane_idx == 0 && generate_consensus){
        uint8_t* consensus = &consensus_d[window_idx * CUDAPOA_MAX_CONSENSUS_SIZE];
        uint16_t* coverage = &coverage_d[window_idx * CUDAPOA_MAX_CONSENSUS_SIZE];
        int32_t* consensus_scores = &consensus_scores_d[window_idx * max_nodes_per_window];
        int16_t* consensus_predecessors = &consensus_predecessors_d[window_idx * max_nodes_per_window];

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
    }

}

// Host function call for POA kernel.
void generatePOA(genomeworks::cudapoa::OutputDetails * output_details_d,
                genomeworks::cudapoa::InputDetails * input_details_d,
                uint32_t total_windows,
                cudaStream_t stream,
                genomeworks::cudapoa::AlignmentDetails * alignment_details_d,
                genomeworks::cudapoa::GraphDetails * graph_details_d,
                int16_t gap_score,
                int16_t mismatch_score,
                int16_t match_score,
                bool cuda_banded_alignment)
{
    // unpack output details
    uint8_t* consensus_d = output_details_d->consensus;
    uint16_t* coverage_d = output_details_d->coverage;
    // unpack input details
    uint8_t* sequences_d = input_details_d->sequences;
    uint8_t* base_weights_d = input_details_d->base_weights;
    uint16_t* sequence_lengths_d = input_details_d->sequence_lengths;
    WindowDetails* window_details_d = input_details_d->window_details;
    // unpack alignment details
    int16_t* scores = alignment_details_d->scores;
    int16_t* alignment_graph = alignment_details_d->alignment_graph;
    int16_t* alignment_read = alignment_details_d->alignment_read;
    // unpack graph details
    uint8_t* nodes = graph_details_d->nodes;
    uint16_t* node_alignments = graph_details_d->node_alignments;
    uint16_t* node_alignment_count = graph_details_d->node_alignment_count;
    uint16_t* incoming_edges = graph_details_d->incoming_edges;
    uint16_t* incoming_edge_count = graph_details_d->incoming_edge_count;
    uint16_t* outgoing_edges = graph_details_d->outgoing_edges;
    uint16_t* outgoing_edge_count = graph_details_d->outgoing_edge_count;
    uint16_t* incoming_edge_w = graph_details_d->incoming_edge_weights;
    uint16_t* outgoing_edge_w = graph_details_d->outgoing_edge_weights;
    uint16_t* sorted_poa = graph_details_d->sorted_poa;
    uint16_t* node_id_to_pos = graph_details_d->sorted_poa_node_map;
    uint16_t* sorted_poa_local_edge_count = graph_details_d->sorted_poa_local_edge_count;
    int32_t* consensus_scores = graph_details_d->consensus_scores;
    int16_t* consensus_predecessors = graph_details_d->consensus_predecessors;
    uint8_t* node_marks = graph_details_d->node_marks;
    bool* check_aligned_nodes = graph_details_d->check_aligned_nodes;
    uint16_t* nodes_to_visit = graph_details_d->nodes_to_visit;
    uint16_t* node_coverage_counts = graph_details_d->node_coverage_counts;
    

    uint32_t nwindows_per_block = CUDAPOA_THREADS_PER_BLOCK/WARP_SIZE;
    uint32_t nblocks = (total_windows + nwindows_per_block - 1)/nwindows_per_block;

    GW_CU_CHECK_ERR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

    if (cuda_banded_alignment) {
        generatePOAKernel<CUDAPOA_BANDED_THREADS_PER_BLOCK, true>
                <<<total_windows, CUDAPOA_BANDED_THREADS_PER_BLOCK, 0, stream>>> (consensus_d,
                                                                                  coverage_d,
                                                                                  sequences_d,
                                                                                  base_weights_d,
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
                                                                                  node_coverage_counts,
                                                                                  gap_score,
                                                                                  mismatch_score,
                                                                                  match_score);
    } else{
        generatePOAKernel<CUDAPOA_THREADS_PER_BLOCK, false>
                <<<nblocks, CUDAPOA_THREADS_PER_BLOCK, 0, stream>>> (consensus_d,
                                                                     coverage_d,
                                                                     sequences_d,
                                                                     base_weights_d,
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
                                                                     node_coverage_counts,
                                                                     gap_score,
                                                                     mismatch_score,
                                                                     match_score);
    }
}

} // namespace cudapoa

} // namespace genomeworks
