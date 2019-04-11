#include "cudapoa_kernels.cuh"
#include <stdio.h>

#define WARP_SIZE 32

namespace nvidia {

namespace cudapoa {

/**
 * @brief Device function for running Needleman-Wunsch dynamic programming loop.
 *
 * @param[in] nodes                Device buffer with unique nodes in graph
 * @param[in] graph                Device buffer with sorted graph
 * @param[in] node_id_to_pos       Device scratch space for mapping node ID to position in graph
 * @param[in] incoming_edge_count  Device buffer with number of incoming edges per node
 * @param[in] incoming_edges       Device buffer with incoming edges per node
 * @param[in] outgoing_edge_count  Device buffer with number of outgoing edges per node
 * @param[in] outgoing_edges       Device buffer with outgoing edges per node
 * @param[in] read                 Device buffer with sequence (read) to align
 * @param[in] read_count           Number of bases in read
 * @param[out] scores              Device scratch space that scores alignment matrix score
 * @param[out] alignment_graph     Device scratch space for backtrace alignment of graph
 * @param[out] alignment_read      Device scratch space for backtrace alignment of sequence
 *
 * @return Number of nodes in final alignment.
 */
__device__
uint16_t runNeedlemanWunsch(uint8_t* nodes,
                        uint16_t* graph,
                        uint16_t* node_id_to_pos,
                        uint16_t graph_count,
                        uint16_t* incoming_edge_count,
                        uint16_t* incoming_edges,
                        uint16_t* outgoing_edge_count,
                        uint16_t* outgoing_edges,
                        uint8_t* read,
                        uint16_t read_count,
                        int16_t* scores,
                        int16_t* alignment_graph,
                        int16_t* alignment_read)
{
    //printf("Running NW\n");
    // Set gap/mismatch penalty. Currently acquired from default racon settings.
    // TODO: Pass scores from arguments.
#pragma message("TODO: Pass match/gap/mismatch scores into NW kernel as parameters.")
    const int16_t GAP = -8;
    const int16_t MISMATCH = -6;
    const int16_t MATCH = 8;

    __shared__ int16_t first_element_prev_score;

    uint32_t thread_idx = threadIdx.x;
    uint32_t warp_idx = thread_idx / WARP_SIZE;

    long long int start = clock64();
    long long int init = 0;

    //for(uint16_t graph_pos = thread_idx; graph_pos < graph_count; graph_pos += blockDim.x)
    //{
    //    //node_id_to_pos[graph_pos] = node_id_to_pos_global[graph_pos];
    //    //incoming_edge_count[graph_pos] = incoming_edge_count_global[graph_pos];
    //}

    // Init horizonal boundary conditions (read).
    for(uint16_t j = thread_idx + 1; j < read_count + 1; j += blockDim.x)
    {
        //score_prev_i[j] = j * GAP;
        scores[j] = j * GAP;
    }

    if (thread_idx == 0)
    {
#ifdef DEBUG
        printf("graph %d, read %d\n", graph_count, read_count);
#endif

        // Init vertical boundary (graph).
        for(uint16_t graph_pos = 0; graph_pos < graph_count; graph_pos++)
        {
            //node_id_to_pos[graph_pos] = node_id_to_pos_global[graph_pos];
            //incoming_edge_count[graph_pos] = incoming_edge_count_global[graph_pos];
            //outgoing_edge_count[graph_pos]= outgoing_edge_count[graph_pos];
            //nodes[graph_pos] = nodes_global[graph_pos];

            scores[0] = 0;
            uint16_t node_id = graph[graph_pos];
            uint16_t i = graph_pos + 1;
            //uint16_t pred_count = incoming_edge_count_global[node_id];
            uint16_t pred_count = incoming_edge_count[node_id];
            if (pred_count == 0)
            {
                scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION] = GAP;
            }
            else
            {
                int16_t penalty = SHRT_MIN;
                for(uint16_t p = 0; p < pred_count; p++)
                {
                    uint16_t pred_node_id = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
                    //uint16_t pred_node_graph_pos = node_id_to_pos_global[pred_node_id] + 1;
                    uint16_t pred_node_graph_pos = node_id_to_pos[pred_node_id] + 1;
                    //printf("pred score %d at pos %d\n", 
                    //        scores[pred_node_graph_pos * CUDAPOA_MAX_MATRIX_DIMENSION],
                    //        pred_node_graph_pos);
                    //printf("node id %d parent id %d\n", node_id, pred_node_id);
                    penalty = max(penalty, scores[pred_node_graph_pos * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION]);
                }
                scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION] = penalty + GAP;
            }

            //printf("%d \n", scores[i * CUDAPOA_MAX_MATRIX_DIMENSION]);
            //printf("node %c, score %d\n", nodes[node_id], scores[(graph_pos+1) * CUDAPOA_MAX_MATRIX_DIMENSION]);
        }

        //score_prev_i[0] = 0;
        //for(uint16_t j = 1; j < read_count + 1; j++)
        //{
        //    //printf("%d ", scores[j]);
        //}
        //printf("\n");

        init = clock64() - start;

    }

    __syncthreads();


    //for(uint32_t i = 0; i < graph_count; i++)
    //{
    //    printf("node-%d pos %d %d %d, ", i, /*node_id_to_pos[i],*/
    //            scores[(node_id_to_pos[i] + 1) * CUDAPOA_MAX_MATRIX_DIMENSION],
    //            incoming_edge_count[i],
    //            outgoing_edge_count[i]);
    //    for(uint16_t j  = 0; j < incoming_edge_count[i]; j++)
    //    {
    //        printf("%d ", incoming_edges[i * CUDAPOA_MAX_NODE_EDGES + j]);
    //    }
    //    printf(", ");
    //    for(uint16_t j  = 0; j < outgoing_edge_count[i]; j++)
    //    {
    //        printf("%d ", outgoing_edges[i * CUDAPOA_MAX_NODE_EDGES + j]);
    //    }
    //    printf("\n");
    //}
    //for(uint32_t i = 0; i < read_count + 1; i++)
    //{
    //    printf("%d ", scores[i]);
    //}


    start = clock64();

    long long int serial = 0;

    const uint32_t cells_per_thread = 4;

    // Maximum cols is the first multiple of (cells_per_thread * tb size) that is larger
    // than the read count.
    uint16_t max_cols = (((read_count - 1) / (blockDim.x * cells_per_thread)) + 1) * (blockDim.x * cells_per_thread);
    //if (thread_idx == 0)
    //{
    //    printf("read count %d max cols %d\n", read_count, max_cols);
    //    //printf("row %d\n", i);
    //}

    // Maximum warps is total number of warps needed (based on fixed warp size and cells per thread)
    // to cover the full read. This number is <= max_cols.
    uint16_t max_warps = (((read_count - 1) / (WARP_SIZE * cells_per_thread)) + 1);

    // Run DP loop for calculating scores. Process each row at a time, and
    // compute vertical and diagonal values in parallel.
    for(uint16_t graph_pos = 0; graph_pos < graph_count; graph_pos++)
    {
        uint16_t node_id = graph[graph_pos];
        uint16_t i = graph_pos + 1;

        if (thread_idx == 0)
        {
            first_element_prev_score = scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION];
        }
        uint16_t out_edge_count = outgoing_edge_count[node_id];

        uint16_t pred_count = incoming_edge_count[node_id];

        uint16_t pred_i_1 = (pred_count == 0 ? 0 :
                node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1);
        int16_t* scores_pred_i_1 = &scores[pred_i_1 * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION];

        uint8_t n = nodes[node_id];

        // max_cols is the first tb boundary multiple beyond read_count. This is done
        // so all threads in the block enter the loop. The loop has syncthreads, so if
        // any of the threads don't enter, then it'll cause a lock in the system.
        for(uint16_t read_pos = thread_idx * cells_per_thread; read_pos < max_cols; read_pos += blockDim.x * cells_per_thread)
        {
            int16_t score0, score1, score2, score3;
            uint16_t j0, j1, j2, j3;

            // To avoid doing extra work, we clip the extra warps that go beyond the read count.
            // Warp clipping hasn't shown to help too much yet, but might if we increase the tb
            // size in the future.
            if (warp_idx < max_warps)
            {
                //printf("updating vertical for pos %d thread %d\n", read_pos, thread_idx);
                int16_t char_profile0 = (n == read[read_pos + 0] ? MATCH : MISMATCH);
                int16_t char_profile1 = (n == read[read_pos + 1] ? MATCH : MISMATCH);
                int16_t char_profile2 = (n == read[read_pos + 2] ? MATCH : MISMATCH);
                int16_t char_profile3 = (n == read[read_pos + 3] ? MATCH : MISMATCH);
                // Index into score matrix.
                j0 = read_pos + 1;
                j1 = read_pos + 2;
                j2 = read_pos + 3;
                j3 = read_pos + 4;
                //printf("thread idx %d locations %d %d\n", thread_idx, j0, j1);
                score0 = max(scores_pred_i_1[j0-1] + char_profile0,
                        scores_pred_i_1[j0] + GAP);
                score1 = max(scores_pred_i_1[j1-1] + char_profile1,
                        scores_pred_i_1[j1] + GAP);
                score2 = max(scores_pred_i_1[j2-1] + char_profile2,
                        scores_pred_i_1[j2] + GAP);
                score3 = max(scores_pred_i_1[j3-1] + char_profile3,
                        scores_pred_i_1[j3] + GAP);

                // Perform same score updates as above, but for rest of predecessors.
                for (uint16_t p = 1; p < pred_count; p++)
                {
                    int16_t pred_i_2 = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1;
                    int16_t* scores_pred_i_2 = &scores[pred_i_2 * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION];

                    score0 = max(scores_pred_i_2[j0 - 1] + char_profile0,
                            max(score0, scores_pred_i_2[j0] + GAP));

                    score1 = max(scores_pred_i_2[j1 - 1] + char_profile1,
                            max(score1, scores_pred_i_2[j1] + GAP));

                    score2 = max(scores_pred_i_2[j2 - 1] + char_profile2,
                            max(score2, scores_pred_i_2[j2] + GAP));

                    score3 = max(scores_pred_i_2[j3 - 1] + char_profile3,
                            max(score3, scores_pred_i_2[j3] + GAP));
                }
            }

            long long int temp = clock64();

            // Process the warps in a step ladder fashion.
            // warp0, warp1, warp2, warp3
            //        warp1, warp2, warp3
            //               warp2, warp3
            //                      warp3
            // After each step, the maximum values are passed onto
            // warp through shared memory.
            // Empirically, tb = 64 works best (2 warps), as increasing beyond that
            // incurs high syncthreads costs that don't offset the hain in parallelism.
            for(uint32_t tb_start = 0; tb_start < blockDim.x; tb_start += WARP_SIZE)
            {
                if (thread_idx >= tb_start && warp_idx < max_warps)
                {
                    int16_t last_score;
                    // While there are changes to the horizontal score values, keep updating the matrix.
                    // So loop will only run the number of time there are corrections in the matrix.
                    // The any_sync warp primitive lets us easily check if any of the threads had an update.
                    bool loop = true;
                    while(__any_sync(0xffffffff, loop))
                    {

                        // To increase instruction level parallelism, we compute the scores
                        // in reverse order (score3 first, then score2, then score1, etc). 
                        // And then check if any of the scores had an update,
                        // and if there's an update then we rerun the loop to capture the effects
                        // of the change in the next loop.
                        loop = false;

                        // The shfl_up lets us grab a value from the lane below.
                        last_score = __shfl_up_sync(0xffffffff << 1, score3, 1);
                        if (thread_idx % 32 == 0)
                        {
                            last_score = score0;
                        }
                        if (thread_idx == tb_start)
                        {
                            last_score = first_element_prev_score;
                        }
                        __syncwarp();

                        bool check3 = false;
                        int16_t tscore = max(score2 + GAP, score3);
                        if (tscore > score3)
                        {
                            score3 = tscore;
                            check3 = true;
                        }

                        bool check2 = false;
                        tscore = max(score1 + GAP, score2);
                        if (tscore > score2)
                        {
                            score2 = tscore;
                            check2 = true;
                        }

                        bool check1 = false;
                        tscore = max(score0 + GAP, score1);
                        if (tscore > score1)
                        {
                            score1 = tscore;
                            check1 = true;
                        }

                        bool check0 = false;
                        tscore = max(last_score + GAP, score0);
                        if (tscore > score0)
                        {
                            score0 = tscore;
                            check0 = true;
                        }
                        //TODO: See if using only one `check` variable affects performance.
                        loop = check0 || check1 || check2 || check3;
                    }

                    // Copy over the last element score of the last lane into a register of first lane
                    // which can be used to compute the first cell of the next warp.
                    if (thread_idx == tb_start + (WARP_SIZE - 1))
                    {
                        first_element_prev_score = score3;
                    }
                }

                //if (thread_idx == 0)
                //{
                //    printf("previous score for thread %d is %d\n", thread_idx, prev_score);
                //}

                __syncthreads();
            }

            // Index into score matrix.
            if (warp_idx < max_warps)
            {
                scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j0] = score0;
                scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j1] = score1;
                scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j2] = score2;
                scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j3] = score3;
            }

            serial += (clock64() - temp);

            __syncthreads();
        }
    }

//    if (thread_idx == 0)
//    {
//        for(uint32_t i = 0; i < graph_count + 1; i++)
//        {
//            for(uint32_t j = 0; j < read_count + 1; j++)
//            {
//                printf("%05d\n", scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j]);
//            }
//        }
//    }

    long long int nw = clock64() - start;
    long long int tb = 0;

    start = clock64();

    uint16_t aligned_nodes = 0;
    if (thread_idx == 0)
    {
        // Find location of the maximum score in the matrix.
        int16_t i = 0;
        int16_t j = read_count;
        int16_t mscore = SHRT_MIN;

        for (int16_t idx = 1; idx <= graph_count; idx++)
        {
            if (outgoing_edge_count[graph[idx - 1]] == 0)
            {
                int16_t s = scores[idx * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j];
                if (mscore < s)
                {
                    mscore = s;
                    i = idx;
                }
            }
        }

        // Fill in backtrace

        int16_t prev_i = 0;
        int16_t prev_j = 0;

        //printf("maxi %d maxj %d score %d\n", i, j, scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j]);

        // Trace back from maximum score position to generate alignment.
        // Trace back is done by re-calculating the score at each cell
        // along the path to see which preceding cell the move could have
        // come from. This seems computaitonally more expensive, but doesn't
        // require storing any traceback buffer during alignment.
        while(!(i == 0 && j == 0))
        {
            //printf("%d %d\n", i, j);
            int16_t scores_ij = scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j];
            bool pred_found = false;
            // printf("%d %d node %d\n", i, j, graph[i-1]);

            // Check if move is diagonal.
            if (i != 0 && j != 0)
            {
                uint16_t node_id = graph[i - 1];
                int16_t match_cost = (nodes[node_id] == read[j-1] ? MATCH : MISMATCH);

                uint16_t pred_count = incoming_edge_count[node_id];
                uint16_t pred_i = (pred_count == 0 ? 0 :
                        (node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1));

                //printf("j %d\n", j-1);
                if (scores_ij == (scores[pred_i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + (j - 1)] + match_cost))
                {
                    prev_i = pred_i;
                    prev_j = j - 1;
                    pred_found = true;
                }

                if (!pred_found)
                {
                    for(uint16_t p = 1; p < pred_count; p++)
                    {
                        pred_i = (node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1);

                        if (scores_ij == (scores[pred_i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + (j - 1)] + match_cost))
                        {
                            prev_i = pred_i;
                            prev_j = j - 1;
                            pred_found = true;
                            break;
                        }
                    }
                }
            }

            // Check if move is vertical.
            if (!pred_found && i != 0)
            {
                uint16_t node_id = graph[i - 1];
                uint16_t pred_count = incoming_edge_count[node_id];
                uint16_t pred_i = (pred_count == 0 ? 0 :
                        node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1);

                if (scores_ij == scores[pred_i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j] + GAP)
                {
                    prev_i = pred_i;
                    prev_j = j;
                    pred_found = true;
                }

                if (!pred_found)
                {
                    for(uint16_t p = 1; p < pred_count; p++)
                    {
                        pred_i = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1;

                        if (scores_ij == scores[pred_i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + j] + GAP)
                        {
                            prev_i = pred_i;
                            prev_j = j;
                            pred_found = true;
                            break;
                        }
                    }
                }
            }

            // Check if move is horizontal.
            if (!pred_found && scores_ij == scores[i * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION + (j - 1)] + GAP)
            {
                prev_i = i;
                prev_j = j - 1;
                pred_found = true;
            }

            alignment_graph[aligned_nodes] = (i == prev_i ? -1 : graph[i-1]);
            alignment_read[aligned_nodes] = (j == prev_j ? -1 : j-1);
            aligned_nodes++;

            //printf("%d %d\n", alignment_graph[aligned_nodes - 1], alignment_read[aligned_nodes-1]);

            i = prev_i;
            j = prev_j;

            //printf("loop %d %d\n",i, j);
        }
#ifdef DEBUG
        printf("aligned nodes %d\n", aligned_nodes);
#endif

        tb = clock64() - start;
    }

    //if (thread_idx == 0)
    //{
    //    long long int total = init + nw + tb;
    //    printf("Total time of init is %lf %\n", ((double)init / total) * 100.f);
    //    printf("Total time of serial is %lf %\n", ((double)serial / total) * 100.f);
    //    printf("Total time of nw is %lf %\n", ((double)(nw - serial) / total) * 100.f);
    //    printf("Total time of tb is %lf %\n", ((double)tb / total) * 100.f);
    //}

    return aligned_nodes;
}

}

}

