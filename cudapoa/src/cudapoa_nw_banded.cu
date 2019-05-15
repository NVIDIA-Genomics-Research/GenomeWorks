#include "cudapoa_kernels.cuh"
#include <stdio.h>

#define WARP_SIZE 32
#define CELLS_PER_THREAD 4

// Extract shorts from bit field.
#define EXTRACT_SHORT_FROM_BITFIELD(type, val, pos) (type)((val >> (16 * (pos))) & 0xffff)

namespace genomeworks {

namespace cudapoa {

__device__ uint16_t get_band_start_for_row(uint16_t row_idx, float gradient, uint16_t band_width, uint16_t max_column){

    int16_t  start_pos = uint16_t(row_idx * gradient) - band_width / 2;

    start_pos = max(start_pos, 0);

    int16_t  end_pos = start_pos + band_width;

    if (end_pos > max_column){
        start_pos = max_column - band_width + CELLS_PER_THREAD;
    };

    start_pos = max(start_pos, 0);

    start_pos = start_pos - (start_pos % CELLS_PER_THREAD);


        //check if end pos is exceeding the matrix bounds, this should never happen becuause band centre doesn't
        // exceed max_column - band_width /2
/*    if ((start_pos + band_width > CUDAPOA_MAX_SEQUENCE_SIZE)) {
        if (threadIdx.x == 0) {
            printf("ERROR - overspilling matrix end!\n");
        }
    }*/


    return uint16_t (start_pos);
}

__device__ int16_t * get_score_ptr(int16_t* scores, uint16_t row, uint16_t  column) {
    return &scores[column + row * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION];
};

__device__ void initialize_band(int16_t* scores, uint16_t row, int16_t  value, float gradient, uint16_t band_width, uint16_t max_column) {

    //hacky
    uint16_t band_start = get_band_start_for_row(row, gradient, band_width, max_column);
    uint16_t band_end = band_start + band_width;
    uint16_t right_pad = band_width; // TODO solve this overallocation of memory

    uint16_t initialization_offset = (band_start == 0) ? 1 : band_start;

    for (uint16_t j = threadIdx.x + initialization_offset; j < band_end + right_pad; j += blockDim.x) { //bandwidth actually readlength, trying to sort out left first
        scores[j + row * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION] = value;
    }
};

__device__ int16_t get_score(int16_t* scores, uint16_t row, uint16_t  column, float gradient, uint16_t bandwidth, uint16_t max_column, int16_t out_of_band_score_offset){
    uint16_t band_start = get_band_start_for_row(row, gradient, bandwidth, max_column);
    uint16_t band_end = band_start + bandwidth;

    if (((column > band_end) || (column < band_start)) && column != 0){
        return SHRT_MIN + out_of_band_score_offset;
    }else {
        return *get_score_ptr(scores, row, column);
    }
}


__device__ void set_score(int16_t* scores, uint16_t row, uint16_t  column, int16_t value, float gradient, uint16_t band_width){
    scores[column + row * CUDAPOA_MAX_MATRIX_SEQUENCE_DIMENSION] = value;
}


__device__
uint16_t runNeedlemanWunschBanded(uint8_t* nodes,
                                  uint16_t* graph,
                                  uint16_t* node_id_to_pos,
                                  uint16_t graph_count,
                                  uint16_t* incoming_edge_count,
                                  uint16_t* incoming_edges,
                                  uint16_t* outgoing_edge_count,
                                  uint16_t* outgoing_edges,
                                  uint8_t* read,
                                  uint16_t read_length,
                                  int16_t* scores,
                                  int16_t* alignment_graph,
                                  int16_t* alignment_read,
                                  int16_t gap_score,
                                  int16_t mismatch_score,
                                  int16_t match_score)
{
    __shared__ int16_t first_element_prev_score;

    int16_t min_score_abs = abs(min(min(gap_score, mismatch_score), match_score) - 1);

    //Calculate gradient for the scores matrix
    float gradient = float(read_length + 1) / float(graph_count + 1);

    uint16_t band_width = blockDim.x * CELLS_PER_THREAD;
    uint32_t thread_idx = threadIdx.x;
    uint32_t warp_idx = thread_idx / WARP_SIZE;

    long long int start = clock64();

    // Initialise the horizontal boundary of the score matrix
    for(uint16_t j = thread_idx + 1; j < read_length + 1; j += blockDim.x)
    {
        set_score(scores, 0, j, j * gap_score, gradient, band_width);
    }


    // Initialise the vertical boundary of the score matrix
    if (thread_idx == 0)
    {
#ifdef DEBUG
        printf("graph %d, read %d\n", graph_count, read_length);
#endif

        for(uint16_t graph_pos = 0; graph_pos < graph_count; graph_pos++)
        {

            set_score(scores, 0, 0, 0, gradient, band_width);

            uint16_t node_id = graph[graph_pos];
            uint16_t i = graph_pos + 1;

            uint16_t pred_count = incoming_edge_count[node_id];
            if (pred_count == 0)
            {
                set_score(scores, i, 0, gap_score, gradient, band_width);
            }
            else
            {
                int16_t penalty = SHRT_MIN;
                for(uint16_t p = 0; p < pred_count; p++)
                {
                    uint16_t pred_node_id = incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p];
                    uint16_t pred_node_graph_pos = node_id_to_pos[pred_node_id] + 1;
                    penalty = max(penalty, get_score(scores, pred_node_graph_pos, 0, gradient, band_width, read_length + 1, min_score_abs));
                }
                set_score(scores, i, 0, penalty + gap_score, gradient, band_width);
            }

        }
    }

    __syncthreads();

    start = clock64();

    long long int serial = 0;

    // Maximum warps is total number of warps needed (based on fixed warp size and cells per thread)
    // to cover the full read. This number is <= max_cols.
    uint16_t max_warps = (((read_length - 1) / (WARP_SIZE * CELLS_PER_THREAD)) + 1);

    // compute vertical and diagonal values in parallel.
    for(uint16_t graph_pos = 0; graph_pos < graph_count; graph_pos++)
    {

        uint16_t node_id = graph[graph_pos];
        uint16_t i = graph_pos + 1;

        uint16_t band_start = get_band_start_for_row(i, gradient, band_width, read_length + 1);

        initialize_band(scores, i, SHRT_MIN + min_score_abs, gradient, band_width, read_length + 1);

        if (thread_idx == 0)
        {
            first_element_prev_score = get_score(scores, i, 0, gradient, band_width, read_length + 1, min_score_abs);
        }

        uint16_t pred_count = incoming_edge_count[node_id];

        uint16_t pred_i_1 = (pred_count == 0 ? 0 :
                             node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1);
        int16_t* scores_pred_i_1 = get_score_ptr(scores, pred_i_1, 0);

        uint8_t n = nodes[node_id];

        // max_cols is the first tb boundary multiple beyond read_length. This is done
        // so all threads in the block enter the loop. The loop has syncthreads, so if
        // any of the threads don't enter, then it'll cause a lock in the system.

        uint16_t read_pos = thread_idx * CELLS_PER_THREAD + band_start;
        {

            int16_t score0, score1, score2, score3;
            uint16_t j0, j1, j2, j3;

            // To avoid doing extra work, we clip the extra warps that go beyond the read count.
            // Warp clipping hasn't shown to help too much yet, but might if we increase the tb
            // size in the future.


            if (warp_idx < max_warps)
            {
                int16_t char_profile0 = (n == read[read_pos + 0] ? match_score : mismatch_score);
                int16_t char_profile1 = (n == read[read_pos + 1] ? match_score : mismatch_score);
                int16_t char_profile2 = (n == read[read_pos + 2] ? match_score : mismatch_score);
                int16_t char_profile3 = (n == read[read_pos + 3] ? match_score : mismatch_score);
                // Index into score matrix.
                j0 = read_pos + 1;
                j1 = read_pos + 2;
                j2 = read_pos + 3;
                j3 = read_pos + 4;

                //if(debug){printf("Indices into score matrix are %i %i %i %i\n", j0, j1, j2, j3);}

                // The load instructions typically load data in 4B or 8B chunks.
                // If data is 16b (2B), then a 4B load chunk is loaded into register
                // and the necessary bits are extracted before returning. This wastes cycles
                // as each read of 16b issues a separate load command.
                // Instead it is better to load a 4B or 8B chunk into a register
                // using a single load inst, and then extracting necessary part of
                // of the data using bit arithmatic. Also reduces register count.

                // This loads 8 consecutive bytes (4 shorts).
                int64_t score_pred_i_1_64 = ((int64_t*)&scores_pred_i_1[j0-1])[0];
                //if(debug){printf("Row for predecessor is %i , getting ptr to position %i\n", pred_i_1, j0-1);}
                // Since we read 5 consecutive shorts, we need to load the next
                // chuink of memory as well.
                int64_t score_pred_i_1_64_2 = ((int64_t*)&scores_pred_i_1[j0-1])[1];

                score0 = max(EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_1_64, 0) + char_profile0,
                             EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_1_64, 1) + gap_score);
                score1 = max(EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_1_64, 1)  + char_profile1,
                             EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_1_64, 2) + gap_score);
                score2 = max(EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_1_64, 2)  + char_profile2,
                             EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_1_64, 3) + gap_score);
                score3 = max(EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_1_64, 3)  + char_profile3,
                             EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_1_64_2, 0) + gap_score);

                // Perform same score updates as above, but for rest of predecessors.
                for (uint16_t p = 1; p < pred_count; p++)
                {
                    int16_t pred_i_2 = node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES + p]] + 1;
                    int16_t* scores_pred_i_2 = get_score_ptr(scores, pred_i_2, 0);

                    // Reasoning for 8B preload same as above.
                    int64_t score_pred_i_2_64 = ((int64_t*)&scores_pred_i_2[j0-1])[0];
                    int64_t score_pred_i_2_64_2 = ((int64_t*)&scores_pred_i_2[j0-1])[1];

                    score0 = max(EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_2_64, 0) + char_profile0,
                                 max(score0, EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_2_64, 1) + gap_score));

                    score1 = max(EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_2_64, 1) + char_profile1,
                                 max(score1, EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_2_64, 2) + gap_score));

                    score2 = max(EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_2_64, 2) + char_profile2,
                                 max(score2, EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_2_64, 3) + gap_score));

                    score3 = max(EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_2_64, 3) + char_profile3,
                                 max(score3, EXTRACT_SHORT_FROM_BITFIELD(int16_t, score_pred_i_2_64_2, 0) + gap_score));
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
            // incurs high syncthreads costs that don't offset the gain in parallelism.
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
                        int16_t tscore = max(score2 + gap_score, score3);
                        if (tscore > score3)
                        {
                            score3 = tscore;
                            check3 = true;
                        }

                        bool check2 = false;
                        tscore = max(score1 + gap_score, score2);
                        if (tscore > score2)
                        {
                            score2 = tscore;
                            check2 = true;
                        }

                        bool check1 = false;
                        tscore = max(score0 + gap_score, score1);
                        if (tscore > score1)
                        {
                            score1 = tscore;
                            check1 = true;
                        }

                        bool check0 = false;
                        tscore = max(last_score + gap_score, score0);
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

                __syncthreads();


            }


            // Index into score matrix.
            if (warp_idx < max_warps)
            {
                set_score(scores, i, j0, score0, gradient, band_width);
                set_score(scores, i, j1, score1, gradient, band_width);
                set_score(scores, i, j2, score2, gradient, band_width);
                set_score(scores, i, j3, score3, gradient, band_width);
            }

            serial += (clock64() - temp);

            __syncthreads();
        }
    }

    long long int nw = clock64() - start;
    //long long int tb = 0;

    start = clock64();

    uint16_t aligned_nodes = 0;
    if (thread_idx == 0)
    {
        // Find location of the maximum score in the matrix.
        int16_t i = 0;
        int16_t j = read_length;
        int16_t mscore = SHRT_MIN;

        for (int16_t idx = 1; idx <= graph_count; idx++)
        {
            if (outgoing_edge_count[graph[idx - 1]] == 0)
            {
                int16_t s = get_score(scores, idx, j, gradient, band_width, read_length + 1, min_score_abs);
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

        while(!(i == 0 && j == 0))
        {

            int16_t scores_ij = get_score(scores, i, j, gradient, band_width, read_length + 1, min_score_abs);
            bool pred_found = false;
            // Check if move is diagonal.
            if (i != 0 && j != 0)
            {

                uint16_t node_id = graph[i - 1];
                int16_t match_cost = (nodes[node_id] == read[j-1] ? match_score : mismatch_score);

                uint16_t pred_count = incoming_edge_count[node_id];
                uint16_t pred_i = (pred_count == 0 ? 0 :
                                   (node_id_to_pos[incoming_edges[node_id * CUDAPOA_MAX_NODE_EDGES]] + 1));

                if (scores_ij == (get_score(scores, pred_i, j -1, gradient, band_width, read_length + 1, min_score_abs) + match_cost))
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

                        if (scores_ij == (get_score(scores, pred_i, j - 1, gradient, band_width, read_length + 1, min_score_abs) + match_cost))
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

                if (scores_ij == get_score(scores, pred_i, j, gradient, band_width, read_length + 1, min_score_abs) + gap_score)
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

                        if (scores_ij == get_score(scores, pred_i, j, gradient, band_width, read_length + 1, min_score_abs) + gap_score)
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
            if (!pred_found && scores_ij == get_score(scores, i, j - 1, gradient, band_width, read_length + 1, min_score_abs) + gap_score)
            {
                prev_i = i;
                prev_j = j - 1;
                pred_found = true;
            }

            alignment_graph[aligned_nodes] = (i == prev_i ? -1 : graph[i-1]);
            alignment_read[aligned_nodes] = (j == prev_j ? -1 : j-1);
            aligned_nodes++;

            i = prev_i;
            j = prev_j;

        }


#ifdef DEBUG
        printf("aligned nodes %d\n", aligned_nodes);
#endif
    }

    return aligned_nodes;
}

}

}
