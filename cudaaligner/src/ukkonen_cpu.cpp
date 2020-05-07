/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "ukkonen_cpu.hpp"
#include <claragenomics/utils/mathutils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <limits>
#include <cassert>
#include <algorithm>

namespace claragenomics
{

namespace cudaaligner
{

namespace
{

inline int clamp_add(int i, int j)
{
    assert(i >= 0);
    assert(j >= 0);
    if (std::numeric_limits<int>::max() - i < j)
    {
        return std::numeric_limits<int>::max();
    }
    else
    {
        return i + j;
    }
}

void ukkonen_build_score_matrix_odd(matrix<int>& scores, char const* target, int n, char const* query, int m, int p, int l, int kdmax)
{
    constexpr int max = std::numeric_limits<int>::max() - 1;
    int const bw      = (1 + n - m + 2 * p + 1) / 2;
    for (int kd = 0; kd <= (kdmax - 1) / 2; ++kd)
    {
        int const lmin = abs(2 * kd + 1 - p);
        int const lmax = 2 * kd + 1 <= p ? 2 * (m - p + 2 * kd + 1) + lmin : (2 * std::min(m, n - (2 * kd + 1) + p) + lmin);
        if (lmin + 1 <= l && l < lmax)
        {
            int const rk    = kd;
            int const rl    = l;
            int const j     = kd - (p + l) / 2 + l;
            int const i     = l - j;
            int const diag  = rl - 2 < 0 ? max : scores(rk, rl - 2) + (query[i - 1] == target[j - 1] ? 0 : 1);
            int const left  = rl - 1 < 0 ? max : scores(rk, rl - 1) + 1;
            int const above = rl - 1 < 0 || rk + 1 >= bw ? max : scores(rk + 1, rl - 1) + 1;
            scores(rk, rl)  = min3(diag, left, above);
        }
    }
}

void ukkonen_build_score_matrix_even(matrix<int>& scores, char const* target, int n, char const* query, int m, int p, int l, int kdmax)
{
    constexpr int max = std::numeric_limits<int>::max() - 1;
    for (int kd = 0; kd <= kdmax / 2; ++kd)
    {
        int const lmin = abs(2 * kd - p);
        int const lmax = 2 * kd <= p ? 2 * (m - p + 2 * kd) + lmin : (2 * std::min(m, n - 2 * kd + p) + lmin);
        if (lmin + 1 <= l && l < lmax)
        {
            int const rk    = kd;
            int const rl    = l;
            int const j     = kd - (p + l) / 2 + l;
            int const i     = l - j;
            int const left  = rk - 1 < 0 || rl - 1 < 0 ? max : scores(rk - 1, rl - 1) + 1;
            int const diag  = rl - 2 < 0 ? max : scores(rk, rl - 2) + (query[i - 1] == target[j - 1] ? 0 : 1);
            int const above = rl - 1 < 0 ? max : scores(rk, rl - 1) + 1;
            scores(rk, rl)  = min3(left, diag, above);
        }
    }
}

} // namespace

std::vector<int8_t> ukkonen_backtrace(matrix<int> const& scores, int n, int m, int p)
{
    // Using scoring schema from cudaaligner.hpp
    // Match = 0
    // Mismatch = 1
    // Insertion = 2
    // Deletion = 3

    using std::get;
    constexpr int max = std::numeric_limits<int>::max() - 1;
    std::vector<int8_t> res;

    int i = m - 1;
    int j = n - 1;

    int k, l;
    std::tie(k, l) = to_band_indices(i, j, p);
    int myscore    = scores(k, l);
    while (i > 0 && j > 0)
    {
        char r          = 0;
        std::tie(k, l)  = to_band_indices(i - 1, j, p);
        int const above = k < 0 || k >= scores.num_rows() || l < 0 || l >= scores.num_cols() ? max : scores(k, l);
        std::tie(k, l)  = to_band_indices(i - 1, j - 1, p);
        int const diag  = k < 0 || k >= scores.num_rows() || l < 0 || l >= scores.num_cols() ? max : scores(k, l);
        std::tie(k, l)  = to_band_indices(i, j - 1, p);
        int const left  = k < 0 || k >= scores.num_rows() || l < 0 || l >= scores.num_cols() ? max : scores(k, l);
        if (left + 1 == myscore)
        {
            r       = static_cast<int8_t>(AlignmentState::insertion);
            myscore = left;
            --j;
        }
        else if (above + 1 == myscore)
        {
            r       = static_cast<int8_t>(AlignmentState::deletion);
            myscore = above;
            --i;
        }
        else
        {
            r       = (diag == myscore ? static_cast<int8_t>(AlignmentState::match) : static_cast<int8_t>(AlignmentState::mismatch));
            myscore = diag;
            --i;
            --j;
        }
        res.push_back(r);
    }
    while (i > 0)
    {
        res.push_back(static_cast<int8_t>(AlignmentState::deletion));
        --i;
    }
    while (j > 0)
    {
        res.push_back(static_cast<int8_t>(AlignmentState::insertion));
        --j;
    }
    std::reverse(res.begin(), res.end());
    return res;
}

matrix<int> ukkonen_build_score_matrix(std::string const& target, std::string const& query, int p)
{
    constexpr int max = std::numeric_limits<int>::max() - 1;
    assert(target.size() >= query.size());
    int const n = target.size() + 1;
    int const m = query.size() + 1;

    int const bw = (1 + n - m + 2 * p + 1) / 2;

    matrix<int> scores(bw, n + m, max);
    scores(0, 0) = 0;
    for (int i = 0; i <= p; ++i)
    {
        int k, l;
        std::tie(k, l) = to_band_indices(i, 0, p);
        scores(k, l)   = i;
    }
    for (int j = 0; j <= (n - m) + p; ++j)
    {
        int k, l;
        std::tie(k, l) = to_band_indices(0, j, p);
        scores(k, l)   = j;
    }

    // Transform to diagonal coordinates
    // (i,j) -> (k=j-i, l=(j+i)/2)
    // where
    // -p <= k <= (n-m)+p
    // abs(k)/2 <= l < (k <= 0 ? m+k : min(m,n-k)
    // shift by p: kd = (k + p)/2, (k + p)/2+1
    int const kdmax = (n - m) + 2 * p;
    for (int lx = 0; lx < n + m; ++lx)
    {
        if (p % 2 == 0)
        {
            ukkonen_build_score_matrix_even(scores, target.c_str(), n, query.c_str(), m, p, 2 * lx, kdmax);
            ukkonen_build_score_matrix_odd(scores, target.c_str(), n, query.c_str(), m, p, 2 * lx + 1, kdmax);
        }
        else
        {
            ukkonen_build_score_matrix_odd(scores, target.c_str(), n, query.c_str(), m, p, 2 * lx, kdmax);
            ukkonen_build_score_matrix_even(scores, target.c_str(), n, query.c_str(), m, p, 2 * lx + 1, kdmax);
        }
    }
    return scores;
}

matrix<int> ukkonen_build_score_matrix_naive(std::string const& target, std::string const& query, int t)
{
    int const n = get_size<int>(target) + 1;
    int const m = get_size<int>(query) + 1;

    int const p = (t - abs(n - m)) / 2;

    matrix<int> scores(m, n, std::numeric_limits<int>::max());
    scores(0, 0) = 0;
    for (int i = 0; i < m; ++i)
        scores(i, 0) = i;
    for (int j = 0; j < n; ++j)
        scores(0, j) = j;

    if (m < n)
    {
        for (int i = 1; i < m; ++i)
        {
            for (int j = 1; j < n; ++j)
            {
                if (-p <= j - i && j - i <= n - m + p)
                    scores(i, j) = min3(
                        clamp_add(scores(i - 1, j), 1),
                        clamp_add(scores(i, j - 1), 1),
                        clamp_add(scores(i - 1, j - 1), (query[i - 1] == target[j - 1] ? 0 : 1)));
            }
        }
    }
    else
    {
        for (int i = 1; i < m; ++i)
        {
            for (int j = 1; j < n; ++j)
            {
                if (-p - (m - n) <= j - i && j - i <= p)
                    scores(i, j) = min3(
                        clamp_add(scores(i - 1, j), 1),
                        clamp_add(scores(i, j - 1), 1),
                        clamp_add(scores(i - 1, j - 1), (query[i - 1] == target[j - 1] ? 0 : 1)));
            }
        }
    }
    return scores;
}

std::vector<int8_t> ukkonen_cpu(std::string const& target, std::string const& query, int const p)
{
    int const n        = target.size() + 1;
    int const m        = query.size() + 1;
    matrix<int> scores = ukkonen_build_score_matrix(target, query, p);
    std::vector<int8_t> result;
    result = ukkonen_backtrace(scores, n, m, p);
    return result;
}

} // namespace cudaaligner
} // namespace claragenomics
