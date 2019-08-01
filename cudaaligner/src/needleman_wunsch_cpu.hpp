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

#include "matrix_cpu.hpp"

#include <vector>
#include <string>

namespace claragenomics
{

namespace cudaaligner
{

matrix<int> needleman_wunsch_build_score_matrix_naive(std::string const& text, std::string const& query);

std::vector<int8_t> needleman_wunsch_cpu(std::string const& text, std::string const& query);

} // namespace cudaaligner
} // namespace claragenomics
