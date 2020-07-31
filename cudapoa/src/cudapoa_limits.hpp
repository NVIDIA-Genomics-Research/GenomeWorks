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

#include <stdint.h>

#include <stdio.h>

#include <claraparabricks/genomeworks/cudapoa/batch.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

static bool use32bitScore(const BatchConfig& batch_size, const int16_t gap_score, const int16_t mismatch_score, const int16_t match_score)
{
    // theoretical max score takes place when sequence and graph completely match with each other
    int32_t upper_bound = batch_size.max_sequence_size * match_score;
    // theoretical min score takes place when sequence and graph do not include a single match
    // it is assumed max_sequence_size <= max_num_nodes; gap_score and match_scores are negative, and match_score is positive
    int32_t max_num_nodes = batch_size.max_nodes_per_graph;
    int32_t lower_bound   = batch_size.max_sequence_size * std::max(gap_score, mismatch_score) + (max_num_nodes - batch_size.max_sequence_size) * gap_score;
    // if theoretical upper or lower bound exceed the range represented by int16_t, then int32_t should be used
    return (upper_bound > INT16_MAX || (-lower_bound) > (INT16_MAX + 1));
}

static bool use32bitSize(const BatchConfig& batch_size)
{
    int32_t max_length = batch_size.max_consensus_size;
    max_length         = std::max(max_length, batch_size.max_nodes_per_graph);
    max_length         = std::max(max_length, batch_size.matrix_sequence_dimension);
    //if max array length in POA analysis exceeds the range represented by int16_t, then int32_t should be used
    return (max_length > INT16_MAX);
}

} // namespace cudapoa
} // namespace genomeworks
} // namespace claraparabricks
