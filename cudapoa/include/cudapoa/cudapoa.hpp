/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

/// \defgroup cudapoa CUDA POA package
/// Base docs for the cudapoa package (tbd)
/// \{

namespace claragenomics
{
namespace cudapoa
{
/// CUDA POA error type
enum class StatusType
{
    success = 0,
    exceeded_maximum_poas,
    exceeded_maximum_sequence_size,
    exceeded_maximum_sequences_per_poa,
    exceeded_batch_size,
    node_count_exceeded_maximum_graph_size,
    edge_count_exceeded_maximum_graph_size,
    seq_len_exceeded_maximum_nodes_per_window,
    loop_count_exceeded_upper_bound,
    generic_error
};

StatusType Init();

enum OutputType
{
    consensus = 0x1,
    msa       = 0x1 << 1
};

} // namespace cudapoa
} // namespace claragenomics
/// \}
