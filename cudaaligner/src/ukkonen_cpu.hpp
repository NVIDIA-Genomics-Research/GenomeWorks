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

#include "matrix_cpu.hpp"

#include <claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp>

#include <vector>
#include <tuple>

namespace claraparabricks
{

namespace genomeworks
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

} // namespace genomeworks

} // namespace claraparabricks
