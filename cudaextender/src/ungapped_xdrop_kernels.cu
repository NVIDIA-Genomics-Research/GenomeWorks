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
/*
* This algorithm was adapted from SegAlign's Ungapped Extender authored by
* Sneha Goenka (gsneha@stanford.edu) and Yatish Turakhia (yturakhi@ucsc.edu).
* Source code for original implementation and use in SegAlign can be found
* here: https://github.com/gsneha26/SegAlign
* Description of the algorithm and original implementation can be found in the SegAlign 
* paper published in SC20 (https://doi.ieeecomputersociety.org/10.1109/SC41405.2020.00043)
*/
#include "ungapped_xdrop_kernels.cuh"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{
// extend the seed values to a segment by ungapped x-drop method, adjust low-scoring
// segment scores based on entropy factor, compare resulting segment scores
// to score_threshold and update the d_scored_segment_pairs and d_done vectors
__global__ void find_high_scoring_segment_pairs(const int8_t* __restrict__ d_target,
                                                const int32_t target_length,
                                                const int8_t* __restrict__ d_query,
                                                const int32_t query_length,
                                                const int32_t* d_sub_mat,
                                                const bool no_entropy,
                                                const int32_t xdrop_threshold,
                                                const int32_t score_threshold,
                                                const SeedPair* d_seed_pairs,
                                                const int32_t num_seed_pairs,
                                                const int32_t start_index,
                                                ScoredSegmentPair* d_scored_segment_pairs,
                                                int32_t* d_done)
{
    // TODO - Kernel optimizations -
    // Github Issue: https://github.com/clara-parabricks/GenomeWorks/issues/579

    // TODO - Following variables are an artifact of the hardcoded encoding scheme with a fixed
    // scoring matrix and a fixed alphabet. Will be replaced with cudasequence API.
    // Github Issue: https://github.com/clara-parabricks/GenomeWorks/issues/574
    constexpr int32_t nuc         = 8;
    constexpr int32_t nuc2        = 64;
    constexpr int32_t nuc_entropy = 4;

    constexpr int32_t num_warps = 4;
    constexpr int32_t warp_size = 32;
    const int32_t lane_id       = threadIdx.x % warp_size;
    const int32_t warp_id       = (threadIdx.x - lane_id) / warp_size;
    const float ln_4            = log(4.0f);
    __shared__ int32_t ref_loc[num_warps];
    __shared__ int32_t query_loc[num_warps];
    __shared__ int32_t total_score[num_warps];
    __shared__ int32_t prev_score[num_warps];
    __shared__ int32_t prev_max_score[num_warps];
    __shared__ int32_t prev_max_pos[num_warps];
    __shared__ bool edge_found[num_warps];
    __shared__ bool xdrop_found[num_warps];
    __shared__ bool new_max_found[num_warps];
    __shared__ int32_t left_extent[num_warps];
    __shared__ int32_t extent[num_warps];
    __shared__ int32_t tile[num_warps];
    __shared__ int32_t sub_mat[nuc2];
    // TODO - Accuracy exploration required to check if the following member can be a float and to
    // check the condition for entropy calculation.
    // Github Issue: https://github.com/clara-parabricks/GenomeWorks/issues/575
    __shared__ double entropy[num_warps];

    if (threadIdx.x < nuc2)
    {
        sub_mat[threadIdx.x] = d_sub_mat[threadIdx.x];
    }
    __syncthreads();

    for (int32_t hid0 = blockIdx.x * num_warps; hid0 < num_seed_pairs; hid0 += num_warps * gridDim.x)
    {
        short count[nuc_entropy]     = {0};
        short count_del[nuc_entropy] = {0};
        const int32_t hid            = hid0 + warp_id + start_index;
        if (lane_id == 0)
        {
            if (hid < num_seed_pairs)
            {
                ref_loc[warp_id]   = d_seed_pairs[hid].target_position_in_read;
                query_loc[warp_id] = d_seed_pairs[hid].query_position_in_read;
            }
            else
            {
                ref_loc[warp_id]   = d_seed_pairs[hid0].target_position_in_read;
                query_loc[warp_id] = d_seed_pairs[hid0].query_position_in_read;
            }
            total_score[warp_id]    = 0;
            tile[warp_id]           = 0;
            xdrop_found[warp_id]    = false;
            edge_found[warp_id]     = false;
            new_max_found[warp_id]  = false;
            entropy[warp_id]        = 1.0f;
            prev_score[warp_id]     = 0;
            prev_max_score[warp_id] = 0;
            prev_max_pos[warp_id]   = -1;
            extent[warp_id]         = 0;
        }
        __syncwarp();
        //////////////////////////////////////////////////////////////////
        //Right extension
        while (!xdrop_found[warp_id] && !edge_found[warp_id])
        {
            int32_t thread_score     = 0;
            const int32_t pos_offset = lane_id + tile[warp_id];
            const int32_t ref_pos    = ref_loc[warp_id] + pos_offset;
            const int32_t query_pos  = query_loc[warp_id] + pos_offset;
            int8_t r_chr;
            int8_t q_chr;

            if (ref_pos < target_length && query_pos < query_length)
            {
                r_chr        = d_target[ref_pos];
                q_chr        = d_query[query_pos];
                thread_score = sub_mat[r_chr * nuc + q_chr];
            }
            __syncwarp();

#pragma unroll
            for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
            {
                const int32_t temp_score = __shfl_up_sync(0xFFFFFFFF, thread_score, offset);

                if (lane_id >= offset)
                {
                    thread_score += temp_score;
                }
            }

            thread_score += prev_score[warp_id];

            int32_t max_pos;
            int32_t max_thread_score;

            if (thread_score > prev_max_score[warp_id])
            {
                max_thread_score = thread_score;
                max_pos          = pos_offset;
            }
            else
            {
                max_thread_score = prev_max_score[warp_id];
                max_pos          = prev_max_pos[warp_id];
            }

            __syncwarp();

#pragma unroll
            for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
            {
                const int32_t temp_score = __shfl_up_sync(0xFFFFFFFF, max_thread_score, offset);
                const int32_t temp_pos   = __shfl_up_sync(0xFFFFFFFF, max_pos, offset);

                if (lane_id >= offset)
                {
                    if (temp_score >= max_thread_score)
                    {
                        max_thread_score = temp_score;
                        max_pos          = temp_pos;
                    }
                }
            }

            bool xdrop_done = ((max_thread_score - thread_score) > xdrop_threshold);
            __syncwarp();

#pragma unroll
            for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
            {
                bool temp_xdrop_done = __shfl_up_sync(0xFFFFFFFF, xdrop_done, offset);
                if (lane_id >= offset)
                {
                    xdrop_done |= temp_xdrop_done;
                }
            }

            if (xdrop_done)
            {
                max_thread_score = prev_max_score[warp_id];
                max_pos          = prev_max_pos[warp_id];
            }
            __syncwarp();

#pragma unroll
            for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
            {
                const int32_t temp_score = __shfl_up_sync(0xFFFFFFFF, max_thread_score, offset);
                const int32_t temp_pos   = __shfl_up_sync(0xFFFFFFFF, max_pos, offset);

                if (lane_id >= offset)
                {
                    if (temp_score >= max_thread_score)
                    {
                        max_thread_score = temp_score;
                        max_pos          = temp_pos;
                    }
                }
            }
            __syncwarp();

            if (lane_id == warp_size - 1)
            {
                new_max_found[warp_id] = max_pos > prev_max_pos[warp_id];
                prev_max_pos[warp_id]  = max_pos;
                if (xdrop_done)
                {
                    total_score[warp_id] += max_thread_score;
                    xdrop_found[warp_id] = true;
                    extent[warp_id]      = max_pos;
                    tile[warp_id]        = max_pos;
                }
                else if (ref_pos >= target_length || query_pos >= query_length)
                {
                    total_score[warp_id] += max_thread_score;
                    edge_found[warp_id] = true;
                    extent[warp_id]     = max_pos;
                    tile[warp_id]       = max_pos;
                }
                else
                {
                    prev_score[warp_id]     = thread_score;
                    prev_max_score[warp_id] = max_thread_score;
                    tile[warp_id] += warp_size;
                }
            }
            __syncwarp();

            if (new_max_found[warp_id])
            {
#pragma unroll
                for (int32_t i = 0; i < nuc_entropy; i++)
                {
                    count[i]     = count[i] + count_del[i];
                    count_del[i] = 0;
                }
            }
            __syncwarp();

            if (r_chr == q_chr && r_chr < nuc_entropy)
            {
                if (pos_offset <= prev_max_pos[warp_id])
                {
                    count[r_chr] += 1;
                }
                else
                {
                    count_del[r_chr] += 1;
                }
            }
            __syncwarp();
        }

        __syncwarp();

        ////////////////////////////////////////////////////////////////
        //Left extension

        if (lane_id == 0)
        {
            tile[warp_id]           = 0;
            xdrop_found[warp_id]    = false;
            edge_found[warp_id]     = false;
            new_max_found[warp_id]  = false;
            prev_score[warp_id]     = 0;
            prev_max_score[warp_id] = 0;
            prev_max_pos[warp_id]   = 0;
            left_extent[warp_id]    = 0;
        }

#pragma unroll
        for (int32_t i = 0; i < nuc_entropy; i++)
        {
            count_del[i] = 0;
        }
        __syncwarp();

        while (!xdrop_found[warp_id] && !edge_found[warp_id])
        {
            int32_t thread_score     = 0;
            const int32_t pos_offset = lane_id + 1 + tile[warp_id];
            int8_t r_chr;
            int8_t q_chr;

            if (ref_loc[warp_id] >= pos_offset && query_loc[warp_id] >= pos_offset)
            {
                const int32_t ref_pos   = ref_loc[warp_id] - pos_offset;
                const int32_t query_pos = query_loc[warp_id] - pos_offset;
                r_chr                   = d_target[ref_pos];
                q_chr                   = d_query[query_pos];
                thread_score            = sub_mat[r_chr * nuc + q_chr];
            }

#pragma unroll
            for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
            {
                const int32_t temp_score = __shfl_up_sync(0xFFFFFFFF, thread_score, offset);

                if (lane_id >= offset)
                {
                    thread_score += temp_score;
                }
            }

            thread_score += prev_score[warp_id];

            int32_t max_pos;
            int32_t max_thread_score;

            if (thread_score > prev_max_score[warp_id])
            {
                max_thread_score = thread_score;
                max_pos          = pos_offset;
            }
            else
            {
                max_thread_score = prev_max_score[warp_id];
                max_pos          = prev_max_pos[warp_id];
            }
            __syncwarp();

#pragma unroll
            for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
            {
                const int32_t temp_score = __shfl_up_sync(0xFFFFFFFF, max_thread_score, offset);
                const int32_t temp_pos   = __shfl_up_sync(0xFFFFFFFF, max_pos, offset);

                if (lane_id >= offset)
                {
                    if (temp_score >= max_thread_score)
                    {
                        max_thread_score = temp_score;
                        max_pos          = temp_pos;
                    }
                }
            }

            bool xdrop_done = ((max_thread_score - thread_score) > xdrop_threshold);
            __syncwarp();

#pragma unroll
            for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
            {
                bool temp_xdrop_done = __shfl_up_sync(0xFFFFFFFF, xdrop_done, offset);
                if (lane_id >= offset)
                {
                    xdrop_done |= temp_xdrop_done;
                }
            }

            if (xdrop_done)
            {
                max_thread_score = prev_max_score[warp_id];
                max_pos          = prev_max_pos[warp_id];
            }
            __syncwarp();

#pragma unroll
            for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
            {
                const int32_t temp_score = __shfl_up_sync(0xFFFFFFFF, max_thread_score, offset);
                const int32_t temp_pos   = __shfl_up_sync(0xFFFFFFFF, max_pos, offset);

                if (lane_id >= offset)
                {
                    if (temp_score >= max_thread_score)
                    {
                        max_thread_score = temp_score;
                        max_pos          = temp_pos;
                    }
                }
            }
            __syncwarp();

            if (lane_id == warp_size - 1)
            {
                new_max_found[warp_id] = max_pos > prev_max_pos[warp_id];
                prev_max_pos[warp_id]  = max_pos;
                if (xdrop_done)
                {
                    total_score[warp_id] += max_thread_score;
                    xdrop_found[warp_id] = true;
                    left_extent[warp_id] = max_pos;
                    extent[warp_id] += left_extent[warp_id];
                    tile[warp_id] = max_pos;
                }
                else if (ref_loc[warp_id] < pos_offset || query_loc[warp_id] < pos_offset)
                {
                    total_score[warp_id] += max_thread_score;
                    edge_found[warp_id]  = true;
                    left_extent[warp_id] = max_pos;
                    extent[warp_id] += left_extent[warp_id];
                    tile[warp_id] = max_pos;
                }
                else
                {
                    prev_score[warp_id]     = thread_score;
                    prev_max_score[warp_id] = max_thread_score;
                    tile[warp_id] += warp_size;
                }
            }
            __syncwarp();

            if (new_max_found[warp_id])
            {
#pragma unroll
                for (int32_t i = 0; i < nuc_entropy; i++)
                {
                    count[i]     = count[i] + count_del[i];
                    count_del[i] = 0;
                }
            }
            __syncwarp();

            if (r_chr == q_chr && r_chr < nuc_entropy)
            {
                if (pos_offset <= prev_max_pos[warp_id])
                {
                    count[r_chr] += 1;
                }
                else
                {
                    count_del[r_chr] += 1;
                }
            }
            __syncwarp();
        }

        //////////////////////////////////////////////////////////////////

        if (total_score[warp_id] >= score_threshold && total_score[warp_id] <= 3 * score_threshold && !no_entropy)
        {
            for (int32_t i = 0; i < nuc_entropy; i++)
            {
#pragma unroll
                for (int32_t offset = 1; offset < warp_size; offset = offset << 1)
                {
                    count[i] += __shfl_up_sync(0xFFFFFFFF, count[i], offset);
                }
            }
            __syncwarp();

            if (lane_id == warp_size - 1 && ((count[0] + count[1] + count[2] + count[3]) >= 20))
            {
                double entropy_ln = 0.f;
#pragma unroll
                for (int32_t i = 0; i < nuc_entropy; i++)
                {
                    if (count[i] != 0)
                    {
                        const double probability = static_cast<double>(count[i]) / static_cast<double>(extent[warp_id] + 1);
                        entropy_ln += (probability)*log(probability);
                    }
                }
                entropy[warp_id] = -entropy_ln / ln_4; // Store Entropy with log base 4
            }
        }
        __syncwarp();

        //////////////////////////////////////////////////////////////////

        if (hid < num_seed_pairs)
        {
            if (lane_id == 0)
            {
                if (static_cast<int32_t>(static_cast<double>(total_score[warp_id]) * entropy[warp_id]) >= score_threshold)
                {
                    d_scored_segment_pairs[hid].start_coord.target_position_in_read = ref_loc[warp_id] - left_extent[warp_id];
                    d_scored_segment_pairs[hid].start_coord.query_position_in_read  = query_loc[warp_id] - left_extent[warp_id];
                    d_scored_segment_pairs[hid].length                              = extent[warp_id];
                    if (entropy[warp_id] > 0)
                        d_scored_segment_pairs[hid].score = total_score[warp_id] * entropy[warp_id];
                    d_done[hid - start_index] = 1;
                }
                else
                {
                    d_scored_segment_pairs[hid].start_coord.target_position_in_read = ref_loc[warp_id];
                    d_scored_segment_pairs[hid].start_coord.query_position_in_read  = query_loc[warp_id];
                    d_scored_segment_pairs[hid].length                              = 0;
                    d_scored_segment_pairs[hid].score                               = 0;
                    d_done[hid - start_index]                                       = 0;
                }
            }
        }
        __syncwarp();
    }
}

// Gather the SSPs from the resulting segments to the beginning of the tmp_ssp array
__global__ void compress_output(const int32_t* d_done,
                                const int32_t start_index,
                                const ScoredSegmentPair* d_ssp,
                                ScoredSegmentPair* d_tmp_ssp,
                                const int32_t num_seed_pairs)
{
    const int32_t stride = blockDim.x * gridDim.x;
    const int32_t start  = blockDim.x * blockIdx.x + threadIdx.x;
    for (int32_t id = start; id < num_seed_pairs; id += stride)
    {
        const int32_t reduced_index = d_done[id];
        const int32_t index         = id + start_index;
        if (index > 0)
        {
            if (reduced_index > d_done[index - 1])
            {
                d_tmp_ssp[reduced_index - 1] = d_ssp[index];
            }
        }
        else
        {
            if (reduced_index == 1)
            {
                d_tmp_ssp[0] = d_ssp[start_index];
            }
        }
    }
}

} // namespace cudaextender

} // namespace genomeworks

} // namespace claraparabricks
