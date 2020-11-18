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
*/
#pragma once

#include "ungapped_xdrop.cuh"

#include <claraparabricks/genomeworks/utils/cudautils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

// extend the hits to a segment by ungapped x-drop method, adjust low-scoring
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
                                                const SeedPair* __restrict__ d_seed_pairs,
                                                const int32_t num_seed_pairs,
                                                const int32_t start_index,
                                                ScoredSegmentPair* d_scored_segment_pairs,
                                                int32_t* d_done);

// Gathers the SSPs from the resulting segments to the beginning of the
// tmp_ssp vector
__global__ void compress_output(const int32_t* d_done,
                                const int32_t start_index,
                                const ScoredSegmentPair* d_ssp,
                                ScoredSegmentPair* d_tmp_ssp,
                                int num_hits);

// Binary predicate for sorting the ScoredSegmentPairs
struct scored_segment_pair_equal
{
    __host__ __device__ bool operator()(ScoredSegmentPair x, ScoredSegmentPair y)
    {
            return ( ( 
                        (x.seed_pair.target_position_in_read - x.seed_pair.query_position_in_read) == (y.seed_pair.target_position_in_read - y.seed_pair.query_position_in_read) 
                        ) 

                    &&  

                    ( 
                     ( 
                      (x.seed_pair.target_position_in_read >= y.seed_pair.target_position_in_read) 
                      && 
                      ( (x.seed_pair.target_position_in_read + x.length) <= (y.seed_pair.target_position_in_read + y.length) )  
                      ) 

                     || 

                     ( 
                      ( y.seed_pair.target_position_in_read >= x.seed_pair.target_position_in_read ) 
                      && 
                      ( (y.seed_pair.target_position_in_read + y.length) <= (x.seed_pair.target_position_in_read + x.length) ) 
                      ) 
                     ) 
                    );
    }
};

struct scored_segment_pair_comp
{
    __host__ __device__ bool operator()(const ScoredSegmentPair& x, const ScoredSegmentPair& y)
    {
        if ( (x.seed_pair.target_position_in_read - x.seed_pair.query_position_in_read) < ( y.seed_pair.target_position_in_read - y.seed_pair.query_position_in_read))
            return true;
        else if ( (x.seed_pair.target_position_in_read - x.seed_pair.query_position_in_read) == ( y.seed_pair.target_position_in_read - y.seed_pair.query_position_in_read))
        {
            if (x.seed_pair.target_position_in_read < y.seed_pair.target_position_in_read)
                return true;
            else if (x.seed_pair.target_position_in_read == y.seed_pair.target_position_in_read)
            {
                if (x.seed_pair.query_position_in_read > y.seed_pair.query_position_in_read)
                    return true;
                else if (x.seed_pair.query_position_in_read == y.seed_pair.query_position_in_read)
                {
                    if (x.score > y.score)
                        return true;
                    else
                        return false;
                }
                else
                    return false;
            }
            else
                return false;
        }
        else
            return false;
    }
};

} // namespace cudaextender

} // namespace genomeworks

} // namespace claraparabricks
