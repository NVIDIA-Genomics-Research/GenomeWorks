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

#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

#include <cassert>
#include <climits>
#include <vector>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

using WordType = uint32_t;

int32_t myers_advance_block(WordType hmask, int32_t carry_in, WordType eq, WordType& pv, WordType& mv)
{
    assert((pv & mv) == WordType(0));

    // Stage 1
    WordType xv = eq | mv;
    if (carry_in < 0)
        eq |= WordType(1);
    WordType xh = (((eq & pv) + pv) ^ pv) | eq;
    WordType ph = mv | (~(xh | pv));
    WordType mh = pv & xh;

    int32_t carry_out = ((ph & hmask) == WordType(0) ? 0 : 1) - ((mh & hmask) == WordType(0) ? 0 : 1);

    ph <<= 1;
    mh <<= 1;

    if (carry_in < 0)
        mh |= WordType(1);

    if (carry_in > 0)
        ph |= WordType(1);

    // Stage 2
    pv = mh | (~(xv | ph));
    mv = ph & xv;

    return carry_out;
}

WordType myers_preprocess(char x, std::string const& query, int32_t offset)
{
    assert(offset < get_size(query));
    const int32_t max_i = (std::min)(get_size(query) - offset, static_cast<int64_t>(sizeof(WordType) * CHAR_BIT));
    WordType r          = 0;
    for (int32_t i = 0; i < max_i; ++i)
    {
        if (x == query[i + offset])
            r = r | (WordType(1) << i);
    }
    return r;
}

int32_t myers_compute_edit_distance(std::string const& target, std::string const& query)
{
    constexpr int32_t word_size = sizeof(WordType) * CHAR_BIT;
    const int32_t query_size    = get_size(query);

    if (query_size == 0)
        return get_size(target);

    const int32_t n_words = ceiling_divide(query_size, word_size);

    std::vector<WordType> pv(n_words, ~WordType(0));
    std::vector<WordType> mv(n_words, 0);
    std::vector<int32_t> score(n_words);
    for (int32_t i = 0; i < n_words; ++i)
    {
        score[i] = (std::min)((i + 1) * word_size, query_size);
    }

    for (const char t : target)
    {
        int32_t carry = 0;
        for (int32_t i = 0; i < n_words; ++i)
        {
            const WordType peq_a = myers_preprocess('A', query, i * word_size);
            const WordType peq_c = myers_preprocess('C', query, i * word_size);
            const WordType peq_g = myers_preprocess('G', query, i * word_size);
            const WordType peq_t = myers_preprocess('T', query, i * word_size);
            const WordType hmask = WordType(1) << (i < (n_words - 1) ? word_size - 1 : query_size - (n_words - 1) * word_size - 1);

            const WordType eq = [peq_a, peq_c, peq_g, peq_t](char x) -> WordType {
                assert(x == 'A' || x == 'C' || x == 'G' || x == 'T');
                switch (x)
                {
                case 'A':
                    return peq_a;
                case 'C':
                    return peq_c;
                case 'G':
                    return peq_g;
                case 'T':
                    return peq_t;
                default:
                    return 0;
                }
            }(t);

            carry = myers_advance_block(hmask, carry, eq, pv[i], mv[i]);
            score[i] += carry;
        }
    }
    return score.back();
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
