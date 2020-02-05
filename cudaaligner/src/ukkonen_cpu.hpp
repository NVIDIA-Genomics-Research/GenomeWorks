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

#include <claragenomics/cudaaligner/cudaaligner.hpp>

#include <vector>
#include <tuple>

namespace claragenomics
{

namespace cudaaligner
{

inline std::tuple<int, int> to_band_indices(int i, int j, int p)
{
    int const kd = (j - i + p) / 2;
    int const l  = (j + i);
    return std::make_tuple(kd, l);
}

inline std::tuple<int, int> to_matrix_indices(int kd, int l, int p)
{
    int const j = kd - (p + l) / 2 + l;
    int const i = l - j;
    return std::make_tuple(i, j);
}

std::vector<int8_t> ukkonen_backtrace(matrix<int> const& scores, int n, int m, int p);

matrix<int> ukkonen_build_score_matrix(std::string const& target, std::string const& query, int p);

matrix<int> ukkonen_build_score_matrix_naive(std::string const& target, std::string const& query, int t);

std::vector<int8_t> ukkonen_cpu(std::string const& target, std::string const& query, int const p);

} // namespace cudaaligner
} // namespace claragenomics
