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

#include <stdexcept>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

StatusType Init()
{
    initialize_logger(claraparabricks::genomeworks::logging::LogLevel::warn);
    return StatusType::success;
}

void decode_error(StatusType error_type, std::string& error_message, std::string& error_hint)
{
    switch (error_type)
    {
    case StatusType::exceeded_maximum_poas:
        error_message = "Kernel Error: Number of groups per batch exceeded maximum POAs.";
        error_hint    = "Suggestion  : Evaluate maximum number of groups per batch using BatchBlock::estimate_max_poas().";
        break;
    case StatusType::exceeded_maximum_sequence_size:
        error_message = "Kernel Error: Input read length or output consensus/MSA sequence length exceeded max sequence size.";
        error_hint    = "Suggestion  : Check BatchConfig::max_sequence_size and BatchConfig::max_consensus_size, increase if necessary.";
        break;
    case StatusType::exceeded_maximum_sequences_per_poa:
        error_message = "Kernel Error: Exceeded maximum number of reads per POA.";
        error_hint    = "Suggestion  : Check BatchConfig::max_sequences_per_poa and increase if necessary.";
        break;
    case StatusType::node_count_exceeded_maximum_graph_size:
        error_message = "Kernel Error: Node count exceeded maximum nodes per POA graph.";
        error_hint    = "Suggestion  : Check BatchConfig::max_nodes_per_graph and increase if necessary.";
        break;
    case StatusType::edge_count_exceeded_maximum_graph_size:
        error_message = "Kernel Error: Edge count exceeded maximum edges per graph.";
        error_hint    = "Suggestion  : Check default value of CUDAPOA_MAX_NODE_EDGES, note that increasing this macro would increase memory usage per POA.";
        break;
    case StatusType::exceeded_adaptive_banded_matrix_size:
        error_message = "Kernel Error: Allocated buffer for score/traceback matrix in adaptive banding is not large enough.";
        error_hint    = "Suggestion  : Check BatchConfig::matrix_sequence_dimension and increase if necessary.";
        break;
    case StatusType::exceeded_maximum_predecessor_distance:
        error_message = "Kernel Error: Set value for maximum predecessor distance in Needleman-Wunsch algorithm with traceback buffer is not large enough.";
        error_hint    = "Suggestion  : Check BatchConfig::max_banded_pred_distance and increase if necessary.";
        break;
    case StatusType::loop_count_exceeded_upper_bound:
        error_message = "Kernel Error: Traceback in Needleman-Wunsch algorithm failed.";
        error_hint    = "Suggestion  : You may retry with a different banding mode.";
        break;
    case StatusType::output_type_unavailable:
        error_message = "Kernel Error: Output type not available.";
        error_hint    = "Suggestion  : Check MSA/Consensus selection for output type.";
        break;
    case StatusType::zero_weighted_poa_sequence:
        error_message = "Error      : Input sequence has all zero base weights.";
        error_hint    = "Suggestion : Check base weights of POA group sequences on host.";
        break;
    case StatusType::empty_poa_group:
        error_message = "Error      : No sequence was added to POA group.";
        error_hint    = "Suggestion : Check status vector in Batch::add_poa_group().";
        break;
    case StatusType::generic_error:
        error_message = "Unknown error.";
        error_hint    = "";
        break;
    default:
        throw std::runtime_error("Unknown error type detected.");
    }
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
