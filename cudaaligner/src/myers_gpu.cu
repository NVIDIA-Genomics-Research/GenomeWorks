/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <cassert>
#include <climits>
#include <vector>
#include <utils/signed_integer_utils.hpp>
#include "device_storage.cuh"
#include "myers_gpu.cuh"
#include <cudautils/cudautils.hpp>

namespace claragenomics
{

namespace cudaaligner
{

using WordType = uint32_t;

inline __device__ WordType warp_leftshift_sync(uint32_t warp_mask, WordType v)
{
    constexpr int32_t word_size = sizeof(WordType) * CHAR_BIT;
    const WordType x            = __shfl_up_sync(warp_mask, v >> (word_size - 1), 1);
    v <<= 1;
    if (threadIdx.x != 0)
        v |= x;
    return v;
}

inline __device__ WordType warp_add_sync(uint32_t warp_mask, WordType a, WordType b)
{
    static_assert(sizeof(WordType) == 4);
    static_assert(CHAR_BIT == 8);
    const uint64_t ax = a;
    const uint64_t bx = b;
    uint64_t r        = ax + bx;
    uint32_t carry    = static_cast<uint32_t>(r >> 32);
    r &= 0xffff'ffffull;
    // TODO: I think due to the structure of the Myer blocks,
    // a carry cannot propagate over more than a single block.
    // I.e. a single carry propagation without the loop should be sufficient.
    while (__any_sync(warp_mask, carry))
    {
        uint32_t x = __shfl_up_sync(warp_mask, carry, 1);
        if (threadIdx.x != 0)
            r += x;
        carry = static_cast<uint32_t>(r >> 32);
        r &= 0xffff'ffffull;
    }
    return static_cast<WordType>(r);
}

__device__ int32_t myers_advance_block(uint32_t warp_mask, WordType highest_bit, WordType eq, WordType& pv, WordType& mv, int32_t carry_in)
{
    assert((pv & mv) == WordType(0));

    // Stage 1
    WordType xv = eq | mv;
    if (carry_in < 0)
        eq |= WordType(1);
    WordType xh = warp_add_sync(warp_mask, eq & pv, pv);
    xh          = (xh ^ pv) | eq;
    WordType ph = mv | (~(xh | pv));
    WordType mh = pv & xh;

    int32_t carry_out = ((ph & highest_bit) == WordType(0) ? 0 : 1) - ((mh & highest_bit) == WordType(0) ? 0 : 1);

    ph = warp_leftshift_sync(warp_mask, ph);
    mh = warp_leftshift_sync(warp_mask, mh);

    if (carry_in < 0)
        mh |= WordType(1);

    if (carry_in > 0)
        ph |= WordType(1);

    // Stage 2
    pv = mh | (~(xv | ph));
    mv = ph & xv;

    return carry_out;
}

__device__ WordType myers_preprocess(char x, char const* query, int32_t query_size, int32_t offset)
{
    // Sets a 1 bit at the position of every matching character
    constexpr int32_t word_size = sizeof(WordType) * CHAR_BIT;
    assert(offset < query_size);
    const int32_t max_i = min(query_size - offset, word_size);
    WordType r          = 0;
    for (int32_t i = 0; i < max_i; ++i)
    {
        if (x == query[i + offset])
            r = r | (WordType(1) << i);
    }
    return r;
}

__global__ void myers_compute_edit_distance_kernel(WordType* pv, WordType* mv, int32_t* score, int32_t n_words, char const* target, int32_t target_size, char const* query, int32_t query_size)
{
    constexpr int32_t word_size = sizeof(WordType) * CHAR_BIT;
    constexpr int32_t warp_size = 32;
    assert(warpSize == warp_size);
    assert(threadIdx.x < warp_size);
    assert(query_size != 0);

    for (int32_t idx = threadIdx.x; idx < n_words; idx += warp_size)
    {
        pv[idx]    = ~WordType(0);
        mv[idx]    = 0;
        score[idx] = min((idx + 1) * word_size, query_size);
    }

    char const* const tend = target + target_size;
    for (char const* t = target; t < tend; ++t)
    {
        int32_t warp_carry = 0;
        for (int32_t idx = threadIdx.x; idx < n_words; idx += warp_size)
        {
            const uint32_t warp_mask = idx / warp_size < n_words / warp_size ? 0xffff'ffffu : (1u << (n_words % warp_size)) - 1;

            WordType pv_local = pv[idx];
            WordType mv_local = mv[idx];
            // TODO these might be cached or only computed for the specific t at hand.
            // TODO query load is inefficient
            const WordType peq_a       = myers_preprocess('A', query, query_size, idx * word_size);
            const WordType peq_c       = myers_preprocess('C', query, query_size, idx * word_size);
            const WordType peq_g       = myers_preprocess('G', query, query_size, idx * word_size);
            const WordType peq_t       = myers_preprocess('T', query, query_size, idx * word_size);
            const WordType highest_bit = WordType(1) << (idx == (n_words - 1) ? query_size - (n_words - 1) * word_size - 1 : word_size - 1);

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
            }(*t);

            warp_carry = myers_advance_block(warp_mask, highest_bit, eq, pv_local, mv_local, warp_carry);
            score[idx] += warp_carry;
            if (threadIdx.x == 0)
                warp_carry = 0;
            //            warp_carry = __shfl_down_sync(warp_mask, warp_carry, warp_size - 1);
            if (warp_mask == 0xffff'ffffu)
                warp_carry = __shfl_down_sync(0x8000'0001u, warp_carry, warp_size - 1);
            if (threadIdx.x != 0)
                warp_carry = 0;
            pv[idx] = pv_local;
            mv[idx] = mv_local;
        }
    }
}

int32_t myers_compute_edit_distance(std::string const& target, std::string const& query)
{
    constexpr int32_t warp_size = 32;
    int32_t device_id           = 0;
    constexpr int32_t word_size = sizeof(WordType) * CHAR_BIT;
    if (get_size(query) == 0)
        return get_size(target);

    cudaStream_t stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&stream));

    device_storage<char> target_d(target.size(), device_id);
    device_storage<char> query_d(query.size(), device_id);

    const int32_t n_words = (get_size(query) + word_size - 1) / word_size;
    device_storage<WordType> pv(n_words, device_id);
    device_storage<WordType> mv(n_words, device_id);
    device_storage<int32_t> score(n_words, device_id);

    //    CGA_CU_CHECK_ERR(cudaMemsetAsync(pv.data(), 0, n_words * sizeof(WordType), stream));
    //    CGA_CU_CHECK_ERR(cudaMemsetAsync(mv.data(), 0, n_words * sizeof(WordType), stream));
    //    CGA_CU_CHECK_ERR(cudaMemsetAsync(mv.data(), 0, n_words * sizeof(WordType), stream));

    CGA_CU_CHECK_ERR(cudaMemcpyAsync(target_d.data(), target.data(), sizeof(char) * get_size(target), cudaMemcpyHostToDevice, stream));
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(query_d.data(), query.data(), sizeof(char) * get_size(query), cudaMemcpyHostToDevice, stream));

    myers_compute_edit_distance_kernel<<<1, warp_size, 0, stream>>>(pv.data(), mv.data(), score.data(), n_words, target_d.data(), get_size(target), query_d.data(), get_size(query));

    int32_t result = 0;
    CGA_CU_CHECK_ERR(cudaMemcpyAsync(&result, score.data() + n_words - 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CGA_CU_CHECK_ERR(cudaStreamSynchronize(stream));
    CGA_CU_CHECK_ERR(cudaStreamDestroy(stream));
    return result;
}

} // namespace cudaaligner
} // namespace claragenomics
