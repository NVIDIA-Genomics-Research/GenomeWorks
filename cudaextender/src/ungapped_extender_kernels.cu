/*
* Copyright 2020 NVIDIA CORPORATION.
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
#include "ungapped_xdrop_kernels.cuh"


// extend the hits to a segment by ungapped x-drop method, adjust low-scoring
// segment scores based on entropy factor, compare resulting segment scores
// to hspthresh and update the d_hsp and d_done vectors
__global__
void find_hsps (const char* __restrict__  d_ref_seq, const char* __restrict__  d_query_seq, uint32_t ref_len, uint32_t query_len, int *d_sub_mat, bool noentropy, int xdrop, int hspthresh, int num_hits, seedHit* d_hit, uint32_t start_index, segment* d_hsp, uint32_t* d_done){

    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_size = warpSize;
    int lane_id = thread_id%warp_size;
    int warp_id = (thread_id-lane_id)/warp_size;

    __shared__ uint32_t ref_loc[NUM_WARPS];
    __shared__ uint32_t query_loc[NUM_WARPS];
    __shared__ int total_score[NUM_WARPS];
    __shared__ int prev_score[NUM_WARPS];
    __shared__ int prev_max_score[NUM_WARPS];
    __shared__ uint32_t prev_max_pos[NUM_WARPS];
    __shared__ bool edge_found[NUM_WARPS];
    __shared__ bool xdrop_found[NUM_WARPS];
    __shared__ bool new_max_found[NUM_WARPS];
    __shared__ uint32_t left_extent[NUM_WARPS];
    __shared__ uint32_t extent[NUM_WARPS];
    __shared__ uint32_t tile[NUM_WARPS];
    __shared__ double entropy[NUM_WARPS];

    int thread_score;
    int max_thread_score;
    uint32_t max_pos;
    uint32_t temp_pos;
    bool xdrop_done;
    int temp;
    short count[4];
    short count_del[4];
    char r_chr;
    char q_chr;
    uint32_t ref_pos;
    uint32_t query_pos;
    uint32_t pos_offset;

    __shared__ int sub_mat[NUC2];

    if(thread_id < NUC2){
        sub_mat[thread_id] = d_sub_mat[thread_id];
    }
    __syncthreads();

    for(int hid0 = block_id*NUM_WARPS; hid0 < num_hits; hid0 += NUM_WARPS*num_blocks){
        int hid = hid0 + warp_id + start_index;

        if(hid < num_hits){
            if(lane_id == 0){
                ref_loc[warp_id] = d_hit[hid].ref_start;
                query_loc[warp_id] = d_hit[hid].query_start;
                total_score[warp_id] = 0;
            }
        }
        else{
            if(lane_id == 0){

                ref_loc[warp_id] = d_hit[hid0].ref_start;
                query_loc[warp_id] = d_hit[hid0].query_start;
                total_score[warp_id] = 0;
            }
        }
        __syncwarp();


        //////////////////////////////////////////////////////////////////
        //Right extension

        if(lane_id ==0){
            tile[warp_id] = 0;
            xdrop_found[warp_id] = false;
            edge_found[warp_id] = false;
            new_max_found[warp_id] = false;
            entropy[warp_id] = 1.0f;
            prev_score[warp_id] = 0;
            prev_max_score[warp_id] = -1000;
            prev_max_pos[warp_id] = 0;
            extent[warp_id] = 0;
        }

        count[0] = 0;
        count[1] = 0;
        count[2] = 0;
        count[3] = 0;
        count_del[0] = 0;
        count_del[1] = 0;
        count_del[2] = 0;
        count_del[3] = 0;
        max_pos = 0;

        __syncwarp();

        while(!xdrop_found[warp_id] && !edge_found[warp_id]){
            pos_offset = lane_id + tile[warp_id];
            ref_pos   = ref_loc[warp_id] + pos_offset;
            query_pos = query_loc[warp_id] + pos_offset;
            thread_score = 0;

            if(ref_pos < ref_len && query_pos < query_len){
                r_chr = d_ref_seq[ref_pos];
                q_chr = d_query_seq[query_pos];
                thread_score = sub_mat[r_chr*NUC+q_chr];
            }
            __syncwarp();


#pragma unroll
            for (int offset = 1; offset < warp_size; offset = offset << 1){
                temp = __shfl_up_sync(0xFFFFFFFF, thread_score, offset);

                if(lane_id >= offset){
                    thread_score += temp;
                }
            }


            thread_score += prev_score[warp_id];
            if(thread_score > prev_max_score[warp_id]){
                max_thread_score = thread_score;
                max_pos = pos_offset;
            }
            else{
                max_thread_score = prev_max_score[warp_id];
                max_pos = prev_max_pos[warp_id];
            }

            __syncwarp();

#pragma unroll
            for (int offset = 1; offset < warp_size; offset = offset << 1){
                temp = __shfl_up_sync(0xFFFFFFFF, max_thread_score, offset);
                temp_pos = __shfl_up_sync(0xFFFFFFFF, max_pos, offset);

                if(lane_id >= offset){
                    if(temp >= max_thread_score){
                        max_thread_score = temp;
                        max_pos = temp_pos;
                    }
                }
            }

            xdrop_done = ((max_thread_score-thread_score) > xdrop);
            __syncwarp();

#pragma unroll
            for (int offset = 1; offset < warp_size; offset = offset << 1){
                xdrop_done |= __shfl_up_sync(0xFFFFFFFF, xdrop_done, offset);
            }

            if(lane_id == warp_size-1){

                if(max_pos > prev_max_pos[warp_id])
                    new_max_found[warp_id] = true;
                else
                    new_max_found[warp_id] = false;

                if(xdrop_done){
                    total_score[warp_id] += max_thread_score;
                    xdrop_found[warp_id] = true;
                    extent[warp_id] = max_pos;
                    prev_max_pos[warp_id] = max_pos;
                    tile[warp_id] = max_pos;
                }
                else if(ref_pos >= ref_len || query_pos >= query_len){
                    total_score[warp_id] += max_thread_score;
                    edge_found[warp_id] = true;
                    extent[warp_id] = max_pos;
                    prev_max_pos[warp_id] = max_pos;
                    tile[warp_id] = max_pos;
                }
                else{
                    prev_score[warp_id] = thread_score;
                    prev_max_score[warp_id] = max_thread_score;
                    prev_max_pos[warp_id] = max_pos;
                    tile[warp_id]+= warp_size;
                }
            }
            __syncwarp();

            if(new_max_found[warp_id]){
                for(int i = 0; i < 4; i++){
                    count[i] = count[i] + count_del[i];
                    count_del[i] = 0;
                }
            }
            __syncwarp();

            if(r_chr == q_chr){
                if(pos_offset <= prev_max_pos[warp_id]){
                    count[r_chr] += 1;
                }
                else{
                    count_del[r_chr] += 1;
                }
            }
            __syncwarp();

        }

        __syncwarp();

        ////////////////////////////////////////////////////////////////
        //Left extension

        if(lane_id ==0){
            tile[warp_id] = 0;
            xdrop_found[warp_id] = false;
            edge_found[warp_id] = false;
            new_max_found[warp_id] = false;
            prev_score[warp_id] = 0;
            prev_max_score[warp_id] = -1000;
            prev_max_pos[warp_id] = 0;
            left_extent[warp_id] = 0;
        }

        count_del[0] = 0;
        count_del[1] = 0;
        count_del[2] = 0;
        count_del[3] = 0;
        max_pos = 0;
        __syncwarp();

        while(!xdrop_found[warp_id] && !edge_found[warp_id]){
            pos_offset = lane_id+1+tile[warp_id];
            thread_score = 0;

            if(ref_loc[warp_id] >= pos_offset  && query_loc[warp_id] >= pos_offset){
                ref_pos   = ref_loc[warp_id] - pos_offset;
                query_pos = query_loc[warp_id] - pos_offset;
                r_chr = d_ref_seq[ref_pos];
                q_chr = d_query_seq[query_pos];
                thread_score = sub_mat[r_chr*NUC+q_chr];

            }

#pragma unroll
            for (int offset = 1; offset < warp_size; offset = offset << 1){
                temp = __shfl_up_sync(0xFFFFFFFF, thread_score, offset);

                if(lane_id >= offset){
                    thread_score += temp;
                }
            }

            thread_score += prev_score[warp_id];
            if(thread_score > prev_max_score[warp_id]){
                max_thread_score = thread_score;
                max_pos = pos_offset;
            }
            else{
                max_thread_score = prev_max_score[warp_id];
                max_pos = prev_max_pos[warp_id];
            }
            __syncwarp();

#pragma unroll
            for (int offset = 1; offset < warp_size; offset = offset << 1){
                temp = __shfl_up_sync(0xFFFFFFFF, max_thread_score, offset);
                temp_pos = __shfl_up_sync(0xFFFFFFFF, max_pos, offset);

                if(lane_id >= offset){
                    if(temp >= max_thread_score){
                        max_thread_score = temp;
                        max_pos = temp_pos;
                    }
                }
            }

            xdrop_done = ((max_thread_score-thread_score) > xdrop);
            __syncwarp();

#pragma unroll
            for (int offset = 1; offset < warp_size; offset = offset << 1){
                xdrop_done |= __shfl_up_sync(0xFFFFFFFF, xdrop_done, offset);
            }

            if(lane_id == warp_size-1){

                if(max_pos > prev_max_pos[warp_id])
                    new_max_found[warp_id] = true;
                else
                    new_max_found[warp_id] = false;

                if(xdrop_done){
                    total_score[warp_id]+=max_thread_score;
                    xdrop_found[warp_id] = true;
                    left_extent[warp_id] = max_pos;
                    extent[warp_id] += left_extent[warp_id];
                    prev_max_pos[warp_id] = max_pos;
                    tile[warp_id] = max_pos;
                }
                else if(ref_loc[warp_id] < pos_offset || query_loc[warp_id] < pos_offset){
                    total_score[warp_id]+=max_thread_score;
                    edge_found[warp_id] = true;
                    left_extent[warp_id] = max_pos;
                    extent[warp_id] += left_extent[warp_id];
                    prev_max_pos[warp_id] = max_pos;
                    tile[warp_id] = max_pos;
                }
                else{
                    prev_score[warp_id] = thread_score;
                    prev_max_score[warp_id] = max_thread_score;
                    prev_max_pos[warp_id] = max_pos;
                    tile[warp_id]+=warp_size;
                }
            }
            __syncwarp();

            if(new_max_found[warp_id]){
                for(int i = 0; i < 4; i++){
                    count[i] = count[i] + count_del[i];
                    count_del[i] = 0;
                }
            }
            __syncwarp();

            if(r_chr == q_chr){
                if(pos_offset <= prev_max_pos[warp_id]){
                    count[r_chr] += 1;
                }
                else{
                    count_del[r_chr] += 1;
                }
            }
            __syncwarp();

        }

        //////////////////////////////////////////////////////////////////

        if(total_score[warp_id] >= hspthresh && total_score[warp_id] <= 3*hspthresh && !noentropy){
            for(int i = 0; i < 4; i++){
#pragma unroll
                for (int offset = 1; offset < warp_size; offset = offset << 1){
                    count[i] += __shfl_up_sync(0xFFFFFFFF, count[i], offset);
                }
            }
            __syncwarp();

            if(lane_id == warp_size-1 && ((count[0]+count[1]+count[2]+count[3]) >= 20)){

                entropy[warp_id] = 0.f;
                for(int i = 0; i < 4; i++){
                    entropy[warp_id] += ((double) count[i])/((double) (extent[warp_id]+1)) * ((count[i] != 0) ? log(((double) count[i]) / ((double) (extent[warp_id]+1))): 0.f);
                }
                entropy[warp_id] = -entropy[warp_id]/log(4.0f);
            }
        }
        __syncwarp();

        //////////////////////////////////////////////////////////////////

        if(hid < num_hits){
            if(lane_id == 0){

                if( ((int) (((float) total_score[warp_id])  * entropy[warp_id])) >= hspthresh){
                    d_hsp[hid].ref_start = ref_loc[warp_id] - left_extent[warp_id];
                    d_hsp[hid].query_start = query_loc[warp_id] - left_extent[warp_id];
                    d_hsp[hid].len = extent[warp_id];
                    if(entropy[warp_id] > 0)
                        d_hsp[hid].score = total_score[warp_id]*entropy[warp_id];
                    d_done[hid-start_index] = 1;
                }
                else{
                    d_hsp[hid].ref_start = ref_loc[warp_id];
                    d_hsp[hid].query_start = query_loc[warp_id];
                    d_hsp[hid].len = 0;
                    d_hsp[hid].score = 0;
                    d_done[hid-start_index] = 0;
                }
            }
        }
        __syncwarp();
    }
}