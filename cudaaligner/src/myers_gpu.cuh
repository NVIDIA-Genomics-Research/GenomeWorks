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

#include <cuda_runtime_api.h>
#include "matrix_cpu.hpp"

namespace claragenomics
{
namespace cudaaligner
{

int32_t myers_compute_edit_distance(std::string const& target, std::string const& query);
matrix<int32_t> myers_get_full_score_matrix(std::string const& target, std::string const& query);

void myers_gpu(int8_t* paths_d, int32_t* path_lengths_d, int32_t max_path_length,
               char const* sequences_d,
               int32_t const* sequence_lengths_d,
               int32_t max_target_query_length,
               int32_t n_alignments,
               cudaStream_t stream);

} // end namespace cudaaligner
} // end namespace claragenomics
