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

#include <claraparabricks/genomeworks/cudapoa/cudapoa.hpp>
#include <claraparabricks/genomeworks/logging/logging.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

StatusType Init()
{
    if (logging::LoggingStatus::success != logging::Init())
        return StatusType::generic_error;

    return StatusType::success;
}

std::string decode_error(StatusType error_type)
{
    std::string error_message;
    switch (error_type)
    {
    case genomeworks::cudapoa::StatusType::node_count_exceeded_maximum_graph_size:
        error_message = "Kernel Error:: Node count exceeded maximum nodes per graph in batch";
        break;
    case genomeworks::cudapoa::StatusType::edge_count_exceeded_maximum_graph_size:
        error_message = "Kernel Error:: Edge count exceeded maximum edges per graph in batch";
        break;
    case genomeworks::cudapoa::StatusType::seq_len_exceeded_maximum_nodes_per_window:
        error_message = "Kernel Error:: Sequence length exceeded maximum nodes per window in batch";
        break;
    case genomeworks::cudapoa::StatusType::loop_count_exceeded_upper_bound:
        error_message = "Kernel Error:: Loop count exceeded upper bound in nw algorithm in batch";
        break;
    case genomeworks::cudapoa::StatusType::exceeded_adaptive_banded_matrix_size:
        error_message = "Kernel Error:: Band width set for adaptive matrix allocation is too small in batch";
        break;
    case genomeworks::cudapoa::StatusType::exceeded_maximum_sequence_size:
        error_message = "Kernel Error:: Consensus/MSA sequence size exceeded max sequence size in batch";
        break;
    case genomeworks::cudapoa::StatusType::exceeded_maximum_predecessor_distance:
        error_message = "Kernel Error:: Set value for maximum predecessor distance in traceback NW is too small";
        break;
    default:
        error_message = "Kernel Error:: Unknown error in batch";
        break;
    }
    return error_message;
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
