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

#include <vector>
#include <claragenomics/cudapoa/batch.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/// \brief Create a small set of batch-sizes to increase GPU parallelization.
///        Input POA groups are binned based on their sizes. Similarly sized poa_groups are grouped in the same bin.
///        Separating smaller POA groups from larger ones allows to launch a larger number of groups concurrently.
///        This increase in parallelization translates to faster runtime
///        This multi-batch strategy is particularly useful when input POA groups sizes display a large variance.
///
/// \param list_of_batch_sizes [out]        a set of batch-sizes, covering all input poa_groups
/// \param list_of_groups_per_batch [out]   corresponding POA groups per batch-size bin
/// \param poa_groups [in]                  vector of input poa_groups
/// \param banded_alignment [in]            flag indicating whether banded-alignment or full-alignment is going to be used, default is banded-alignment
/// \param msa_flag [in]                    flag indicating whether MSA or consensus is going to be computed, default is consensus
/// \param bins_capacity [in]               pointer to vector of bins used to create separate different-sized poa_groups, if null as input, a set of default bins will be used
/// \param gpu_memory_usage_quota [in]      portion of GPU available memory that will be used for compute each cudaPOA batch, default 0.9
/// \param mismatch_score [in]              mismatch score, default -6
/// \param gap_score [in]                   gap score, default -8
/// \param match_score [in]                 match core, default 8
void get_multi_batch_sizes(std::vector<BatchSize>& list_of_batch_sizes,
                           std::vector<std::vector<int32_t>>& list_of_groups_per_batch,
                           const std::vector<Group>& poa_groups,
                           bool banded_alignment               = true,
                           bool msa_flag                       = false,
                           int32_t band_width                  = 256,
                           std::vector<int32_t>* bins_capacity = nullptr,
                           float gpu_memory_usage_quota        = 0.9,
                           int32_t mismatch_score              = -6,
                           int32_t gap_score                   = -8,
                           int32_t match_score                 = 8);

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
