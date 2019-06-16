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

#include <cassert>
#include <climits>
#include <utils/signed_integer_utils.hpp>

namespace claragenomics
{

namespace cudaaligner
{

using WordType = uint32_t;

int32_t myers_advance_block(WordType hmask, WordType eq, WordType & pv, WordType & mv)
{
    assert( (pv & mv) == WordType(0) );

    // Stage 1
    WordType xv = eq | mv;
    WordType xh = (((eq & pv) + pv) ^ pv) | eq;
    WordType ph = mv | (~(xh | pv));
    WordType mh = pv & xh;

    int32_t carry = ((ph & hmask) == WordType(0) ? 0 : 1)
       - ((mh & hmask) == WordType(0) ? 0 : 1);

    ph <<= 1;
    mh <<= 1;

    // Stage 2
    pv = mh | (~(xv | ph));
    mv = ph & xv;

    return carry;
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
    std::cout << "q:" << query.size() << std::endl;
    std::cout << "W:" << sizeof(WordType)*CHAR_BIT << std::endl;
    assert( query.size() <= sizeof(WordType)*CHAR_BIT );

    if(get_size(query) == 0)
        return get_size(target);

    const WordType peq_a = myers_preprocess('A', query, 0);
    const WordType peq_c = myers_preprocess('C', query, 0);
    const WordType peq_g = myers_preprocess('G', query, 0);
    const WordType peq_t = myers_preprocess('T', query, 0);

    WordType pv = ~ WordType(0);
    WordType mv =   WordType(0);
    int32_t score = get_size(query);
    const WordType hmask = WordType(1) << (get_size(query) - 1);
    for (const char t : target)
    {
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

        int32_t carry = myers_advance_block(hmask, eq, pv, mv);
        score += carry;
    }
    return score;
}

} // namespace cudaaligner
} // namespace claragenomics
