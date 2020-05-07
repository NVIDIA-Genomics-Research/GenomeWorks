/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "myers_gpu.cuh"
#include "batched_device_matrices.cuh"

#include <claragenomics/cudaaligner/aligner.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/utils/mathutils.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/allocator.hpp>
#include <claragenomics/utils/device_buffer.hpp>

#include <cassert>
#include <climits>
#include <vector>
#include <numeric>

namespace claragenomics
{

namespace cudaaligner
{

constexpr int32_t warp_size = 32;

namespace myers
{

constexpr int32_t word_size                     = sizeof(WordType) * CHAR_BIT;
constexpr int32_t initial_distance_guess_factor = 20;

inline __device__ WordType warp_leftshift_sync(uint32_t warp_mask, WordType v)
{
    assert(((warp_mask >> (threadIdx.x % warp_size)) & 1u) == 1u);
    // 4 threads, word_size = 4 example: thread 0  | thread 1 | thread 2 | thread 3
    // v = 0101 | 0111 | 0011 | 1101 -> 1010 | 1110 | 0111 | 1010
    const WordType x = __shfl_up_sync(warp_mask, v >> (word_size - 1), 1);
    v <<= 1;
    if (threadIdx.x != 0)
        v |= x;
    return v;
}

inline __device__ WordType warp_rightshift_sync(int32_t t, int32_t i, uint32_t warp_mask, WordType v)
{
    assert(((warp_mask >> (threadIdx.x % warp_size)) & 1u) == 1u);
    // 4 threads, word_size = 4 example: thread 0  | thread 1 | thread 2 | thread 3
    // v = 0101 | 0111 | 0011 | 1101 -> 0010 | 1011 | 1001 | 1110
    const WordType x = __shfl_down_sync(warp_mask, v << (word_size - 1), 1);
    v >>= 1;
    if ((warp_mask >> threadIdx.x) > 1u)
        v |= x;
    return v;
}

inline __device__ WordType warp_add_sync(uint32_t warp_mask, WordType a, WordType b)
{
    static_assert(sizeof(WordType) == 4, "This function assumes WordType to have 4 bytes.");
    static_assert(CHAR_BIT == 8, "This function assumes a char width of 8 bit.");
    assert(((warp_mask >> (threadIdx.x % warp_size)) & 1u) == 1u);
    const uint64_t ax = a;
    const uint64_t bx = b;
    uint64_t r        = ax + bx;
    uint32_t carry    = static_cast<uint32_t>(r >> 32);
    if (warp_mask == 1u)
    {
        return static_cast<WordType>(r);
    }
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

__device__ WordType myers_generate_query_pattern(char x, char const* query, int32_t query_size, int32_t offset)
{
    // Sets a 1 bit at the position of every matching character
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

inline __device__ WordType get_query_pattern(device_matrix_view<WordType>& query_patterns, int32_t idx, int32_t query_begin_offset, char x, bool reverse)
{
    static_assert(std::is_unsigned<WordType>::value, "WordType has to be an unsigned type for well-defined >> operations.");
    assert(x >= 0);
    assert(x == 'A' || x == 'C' || x == 'G' || x == 'T');
    const int32_t char_idx = (x >> 1) & 0x3u; // [A,C,T,G] -> [0,1,2,3]

    // 4-bit word example:
    // query_patterns contains character match bit patterns "XXXX" for the full query string.
    // we want the bit pattern "yyyy" for a view of on the query string starting at eg. character 11:
    //       4    3    2     1      0 (pattern index)
    //    XXXX XXXX XXXX [XXXX] [XXXX]
    //     YYY Yyyy y
    //         1    0 (idx)
    //
    // query_begin_offset = 11
    // => idx_offset = 11/4 = 2, shift = 11%4 = 3

    const int32_t idx_offset = query_begin_offset / word_size;
    const int32_t shift      = query_begin_offset % word_size;

    WordType r = query_patterns(idx + idx_offset, char_idx);
    if (shift != 0)
    {
        r >>= shift;
        if (idx + idx_offset + 1 < query_patterns.num_rows())
        {
            r |= query_patterns(idx + idx_offset + 1, char_idx) << (word_size - shift);
        }
    }
    return r;
}

inline __device__ int32_t get_myers_score(int32_t i, int32_t j, device_matrix_view<WordType> const& pv, device_matrix_view<WordType> const& mv, device_matrix_view<int32_t> const& score, WordType last_entry_mask)
{
    assert(i > 0); // row 0 is implicit, NW matrix is shifted by i -> i-1
    const int32_t word_idx = (i - 1) / word_size;
    const int32_t bit_idx  = (i - 1) % word_size;
    int32_t s              = score(word_idx, j);
    WordType mask          = (~WordType(1)) << bit_idx;
    if (word_idx == score.num_rows() - 1)
        mask &= last_entry_mask;
    s -= __popc(mask & pv(word_idx, j));
    s += __popc(mask & mv(word_idx, j));
    return s;
}

__device__ void myers_backtrace(int8_t* paths_base, int32_t* lengths, int32_t max_path_length, device_matrix_view<WordType> const& pv, device_matrix_view<WordType> const& mv, device_matrix_view<int32_t> const& score, int32_t query_size, int32_t id)
{
    using nw_score_t = int32_t;
    assert(pv.num_rows() == score.num_rows());
    assert(mv.num_rows() == score.num_rows());
    assert(pv.num_cols() == score.num_cols());
    assert(mv.num_cols() == score.num_cols());
    assert(score.num_rows() == ceiling_divide(query_size, word_size));
    int32_t i = query_size;
    int32_t j = score.num_cols() - 1;

    int8_t* path = paths_base + id * static_cast<ptrdiff_t>(max_path_length);

    const WordType last_entry_mask = query_size % word_size != 0 ? (WordType(1) << (query_size % word_size)) - 1 : ~WordType(0);

    nw_score_t myscore = score((i - 1) / word_size, j);
    int32_t pos        = 0;
    while (i > 0 && j > 0)
    {
        int8_t r               = 0;
        nw_score_t const above = i == 1 ? j : get_myers_score(i - 1, j, pv, mv, score, last_entry_mask);
        nw_score_t const diag  = i == 1 ? j - 1 : get_myers_score(i - 1, j - 1, pv, mv, score, last_entry_mask);
        nw_score_t const left  = get_myers_score(i, j - 1, pv, mv, score, last_entry_mask);
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
        path[pos] = r;
        ++pos;
    }
    while (i > 0)
    {
        path[pos] = static_cast<int8_t>(AlignmentState::deletion);
        ++pos;
        --i;
    }
    while (j > 0)
    {
        path[pos] = static_cast<int8_t>(AlignmentState::insertion);
        ++pos;
        --j;
    }
    lengths[id] = pos;
}

__global__ void myers_backtrace_kernel(int8_t* paths_base, int32_t* lengths, int32_t max_path_length,
                                       batched_device_matrices<WordType>::device_interface* pvi,
                                       batched_device_matrices<WordType>::device_interface* mvi,
                                       batched_device_matrices<int32_t>::device_interface* scorei,
                                       int32_t const* sequence_lengths_d,
                                       int32_t n_alignments)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_alignments)
        return;
    CGA_CONSTEXPR int32_t word_size         = sizeof(WordType) * CHAR_BIT;
    const int32_t query_size                = sequence_lengths_d[2 * idx];
    const int32_t target_size               = sequence_lengths_d[2 * idx + 1];
    const int32_t n_words                   = (query_size + word_size - 1) / word_size;
    const device_matrix_view<WordType> pv   = pvi->get_matrix_view(idx, n_words, target_size + 1);
    const device_matrix_view<WordType> mv   = mvi->get_matrix_view(idx, n_words, target_size + 1);
    const device_matrix_view<int32_t> score = scorei->get_matrix_view(idx, n_words, target_size + 1);
    myers_backtrace(paths_base, lengths, max_path_length, pv, mv, score, query_size, idx);
}

__global__ void myers_convert_to_full_score_matrix_kernel(batched_device_matrices<int32_t>::device_interface* fullscorei,
                                                          batched_device_matrices<WordType>::device_interface* pvi,
                                                          batched_device_matrices<WordType>::device_interface* mvi,
                                                          batched_device_matrices<int32_t>::device_interface* scorei,
                                                          int32_t const* sequence_lengths_d,
                                                          int32_t alignment)
{
    CGA_CONSTEXPR int32_t word_size = sizeof(WordType) * CHAR_BIT;
    const int32_t query_size        = sequence_lengths_d[2 * alignment];
    const int32_t target_size       = sequence_lengths_d[2 * alignment + 1];
    const int32_t n_words           = (query_size + word_size - 1) / word_size;

    assert(query_size > 0);

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < target_size + 1 && i < query_size + 1)
    {
        const WordType last_entry_mask        = query_size % word_size != 0 ? (WordType(1) << (query_size % word_size)) - 1 : ~WordType(0);
        device_matrix_view<WordType> pv       = pvi->get_matrix_view(0, n_words, target_size + 1);
        device_matrix_view<WordType> mv       = mvi->get_matrix_view(0, n_words, target_size + 1);
        device_matrix_view<int32_t> score     = scorei->get_matrix_view(0, n_words, target_size + 1);
        device_matrix_view<int32_t> fullscore = fullscorei->get_matrix_view(0, query_size + 1, target_size + 1);
        int32_t myscore                       = 0;
        if (i == 0)
            myscore = j;
        else
            myscore = get_myers_score(i, j, pv, mv, score, last_entry_mask);
        fullscore(i, j) = myscore;
    }
}

__global__ void myers_compute_score_matrix_kernel(
    batched_device_matrices<WordType>::device_interface* pvi,
    batched_device_matrices<WordType>::device_interface* mvi,
    batched_device_matrices<int32_t>::device_interface* scorei,
    batched_device_matrices<WordType>::device_interface* query_patternsi,
    char const* sequences_d, int32_t const* sequence_lengths_d,
    int32_t max_sequence_length,
    int32_t n_alignments)
{
    CGA_CONSTEXPR int32_t word_size = sizeof(WordType) * CHAR_BIT;
    CGA_CONSTEXPR int32_t warp_size = 32;
    assert(warpSize == warp_size);
    assert(threadIdx.x < warp_size);
    assert(blockIdx.x == 0);

    const int32_t alignment_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (alignment_idx >= n_alignments)
        return;
    const int32_t query_size        = sequence_lengths_d[2 * alignment_idx];
    const int32_t target_size       = sequence_lengths_d[2 * alignment_idx + 1];
    const char* const query         = sequences_d + 2 * alignment_idx * max_sequence_length;
    const char* const target        = sequences_d + (2 * alignment_idx + 1) * max_sequence_length;
    const int32_t n_words           = (query_size + word_size - 1) / word_size;
    const int32_t n_warp_iterations = ceiling_divide(n_words, warp_size) * warp_size;

    assert(query_size > 0);

    device_matrix_view<WordType> pv             = pvi->get_matrix_view(alignment_idx, n_words, target_size + 1);
    device_matrix_view<WordType> mv             = mvi->get_matrix_view(alignment_idx, n_words, target_size + 1);
    device_matrix_view<int32_t> score           = scorei->get_matrix_view(alignment_idx, n_words, target_size + 1);
    device_matrix_view<WordType> query_patterns = query_patternsi->get_matrix_view(alignment_idx, n_words, 4);

    for (int32_t idx = threadIdx.x; idx < n_words; idx += warp_size)
    {
        pv(idx, 0)    = ~WordType(0);
        mv(idx, 0)    = 0;
        score(idx, 0) = min((idx + 1) * word_size, query_size);
        // TODO query load is inefficient
        query_patterns(idx, 0) = myers_generate_query_pattern('A', query, query_size, idx * word_size);
        query_patterns(idx, 1) = myers_generate_query_pattern('C', query, query_size, idx * word_size);
        query_patterns(idx, 2) = myers_generate_query_pattern('T', query, query_size, idx * word_size);
        query_patterns(idx, 3) = myers_generate_query_pattern('G', query, query_size, idx * word_size);
    }
    __syncwarp();

    for (int32_t t = 1; t <= target_size; ++t)
    {
        int32_t warp_carry = 0;
        if (threadIdx.x == 0)
            warp_carry = 1; // for global alignment the (implicit) first row has to be 0,1,2,3,... -> carry 1
        for (int32_t idx = threadIdx.x; idx < n_warp_iterations; idx += warp_size)
        {
            if (idx < n_words)
            {
                const uint32_t warp_mask = idx / warp_size < n_words / warp_size ? 0xffff'ffffu : (1u << (n_words % warp_size)) - 1;

                WordType pv_local          = pv(idx, t - 1);
                WordType mv_local          = mv(idx, t - 1);
                const WordType highest_bit = WordType(1) << (idx == (n_words - 1) ? query_size - (n_words - 1) * word_size - 1 : word_size - 1);
                const WordType eq          = get_query_pattern(query_patterns, idx, 0, target[t - 1], false);

                warp_carry    = myers_advance_block(warp_mask, highest_bit, eq, pv_local, mv_local, warp_carry);
                score(idx, t) = score(idx, t - 1) + warp_carry;
                if (threadIdx.x == 0)
                    warp_carry = 0;
                if (warp_mask == 0xffff'ffffu && (threadIdx.x == 31 || threadIdx.x == 0))
                    warp_carry = __shfl_down_sync(0x8000'0001u, warp_carry, warp_size - 1);
                if (threadIdx.x != 0)
                    warp_carry = 0;
                pv(idx, t) = pv_local;
                mv(idx, t) = mv_local;
            }
            __syncwarp();
        }
    }
}

__device__ int32_t myers_backtrace_banded(int8_t* path, device_matrix_view<WordType> const& pv, device_matrix_view<WordType> const& mv, device_matrix_view<int32_t> const& score, int32_t diagonal_begin, int32_t diagonal_end, int32_t band_width, int32_t target_size, int32_t query_size)
{
    assert(threadIdx.x == 0);
    using nw_score_t = int32_t;
    assert(pv.num_rows() == score.num_rows());
    assert(mv.num_rows() == score.num_rows());
    assert(pv.num_cols() == score.num_cols());
    assert(mv.num_cols() == score.num_cols());
    assert(score.num_rows() == ceiling_divide(band_width, word_size));
    int32_t i = band_width;
    int32_t j = target_size;

    const WordType last_entry_mask = band_width % word_size != 0 ? (WordType(1) << (band_width % word_size)) - 1 : ~WordType(0);

    nw_score_t myscore = score(i / word_size, j);
    int32_t pos        = 0;
    while (j >= diagonal_end)
    {
        int8_t r               = 0;
        nw_score_t const above = i <= 1 ? j : get_myers_score(i - 1, j, pv, mv, score, last_entry_mask);
        nw_score_t const diag  = i <= 1 ? j - 1 : get_myers_score(i - 1, j - 1, pv, mv, score, last_entry_mask);
        nw_score_t const left  = get_myers_score(i, j - 1, pv, mv, score, last_entry_mask);
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
        path[pos] = r;
        ++pos;
    }
    while (j >= diagonal_begin)
    {
        int8_t r               = 0;
        nw_score_t const above = i <= 1 ? j : get_myers_score(i - 1, j, pv, mv, score, last_entry_mask);
        nw_score_t const diag  = i <= 0 ? j - 1 : get_myers_score(i, j - 1, pv, mv, score, last_entry_mask);
        nw_score_t const left  = get_myers_score(i + 1, j - 1, pv, mv, score, last_entry_mask);
        if (left + 1 == myscore)
        {
            r       = static_cast<int8_t>(AlignmentState::insertion);
            myscore = left;
            ++i;
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
            --j;
        }
        path[pos] = r;
        ++pos;
    }
    while (i > 0 && j > 0)
    {
        int8_t r               = 0;
        nw_score_t const above = i == 1 ? j : get_myers_score(i - 1, j, pv, mv, score, last_entry_mask);
        nw_score_t const diag  = i == 1 ? j - 1 : get_myers_score(i - 1, j - 1, pv, mv, score, last_entry_mask);
        nw_score_t const left  = get_myers_score(i, j - 1, pv, mv, score, last_entry_mask);
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
        path[pos] = r;
        ++pos;
    }
    while (i > 0)
    {
        path[pos] = static_cast<int8_t>(AlignmentState::deletion);
        ++pos;
        --i;
    }
    while (j > 0)
    {
        path[pos] = static_cast<int8_t>(AlignmentState::insertion);
        ++pos;
        --j;
    }
    return pos;
}

__device__ void myers_compute_scores_horizontal_band_impl(
    device_matrix_view<WordType>& pv,
    device_matrix_view<WordType>& mv,
    device_matrix_view<int32_t>& score,
    device_matrix_view<WordType>& query_patterns,
    char const* target_begin,
    char const* query_begin,
    const int32_t target_size,
    const int32_t t_begin,
    const int32_t t_end,
    const int32_t width,
    const int32_t n_words,
    const int32_t pattern_idx_offset)
{
    assert(n_words == ceiling_divide(width, word_size));
    assert(target_size >= 0);
    assert(t_begin <= t_end);
    const int32_t n_warp_iterations = ceiling_divide(n_words, warp_size) * warp_size;
    for (int32_t t = t_begin; t < t_end; ++t)
    {
        int32_t warp_carry = 0;
        if (threadIdx.x == 0)
            warp_carry = 1; // worst case for the top boarder of the band
        for (int32_t idx = threadIdx.x; idx < n_warp_iterations; idx += warp_size)
        {
            if (idx < n_words)
            {
                const uint32_t warp_mask   = idx / warp_size < n_words / warp_size ? 0xffff'ffffu : (1u << (n_words % warp_size)) - 1;
                WordType pv_local          = pv(idx, t - 1);
                WordType mv_local          = mv(idx, t - 1);
                const WordType highest_bit = WordType(1) << (idx == (n_words - 1) ? width - (n_words - 1) * word_size - 1 : word_size - 1);
                const WordType eq          = get_query_pattern(query_patterns, idx, pattern_idx_offset, target_begin[t - 1], false);

                warp_carry    = myers_advance_block(warp_mask, highest_bit, eq, pv_local, mv_local, warp_carry);
                score(idx, t) = score(idx, t - 1) + warp_carry;
                if (threadIdx.x == 0)
                    warp_carry = 0;
                if (warp_mask == 0xffff'ffffu && (threadIdx.x == 0 || threadIdx.x == 31))
                    warp_carry = __shfl_down_sync(0x8000'0001u, warp_carry, warp_size - 1);
                if (threadIdx.x != 0)
                    warp_carry = 0;
                pv(idx, t) = pv_local;
                mv(idx, t) = mv_local;
            }
            __syncwarp();
        }
    }
}

__device__ void myers_compute_scores_diagonal_band_impl(
    device_matrix_view<WordType>& pv,
    device_matrix_view<WordType>& mv,
    device_matrix_view<int32_t>& score,
    device_matrix_view<WordType>& query_patterns,
    char const* target_begin,
    char const* query_begin,
    const int32_t target_size,
    const int32_t t_begin,
    const int32_t t_end,
    const int32_t band_width,
    const int32_t n_words_band,
    const int32_t pattern_idx_offset)
{
    assert(n_words_band == ceiling_divide(band_width, warp_size));
    assert(band_width - (n_words_band - 1) * word_size >= 2); // we need at least two bits in the last word
    const int32_t n_warp_iterations = ceiling_divide(n_words_band, warp_size) * warp_size;
    for (int32_t t = t_begin; t < t_end; ++t)
    {
        int32_t itcnt      = 0;
        int32_t carry_down = 0;
        if (threadIdx.x == 0)
            carry_down = 1; // worst case for the top boarder of the band
        for (int32_t idx = threadIdx.x; idx < n_warp_iterations; idx += warp_size)
        {
            // idx within band column
            const uint32_t warp_mask = idx / warp_size < n_words_band / warp_size ? 0xffff'ffffu : (1u << (n_words_band % warp_size)) - 1;

            if (idx < n_words_band)
            {
                // data from the previous column
                WordType pv_local = warp_rightshift_sync(t, ++itcnt, warp_mask, pv(idx, t - 1));
                WordType mv_local = warp_rightshift_sync(t, ++itcnt, warp_mask, mv(idx, t - 1));
                if (threadIdx.x == 31 && warp_mask == 0xffff'ffffu)
                {
                    if (idx < n_words_band - 1)
                    {
                        pv_local |= pv(idx + 1, t - 1) << (word_size - 1);
                        mv_local |= mv(idx + 1, t - 1) << (word_size - 1);
                    }
                }

                const WordType carry_right_bit = WordType(1) << (idx == (n_words_band - 1) ? band_width - (n_words_band - 1) * word_size - 2 : word_size - 2);
                const WordType carry_down_bit  = carry_right_bit << 1;
                assert(carry_down_bit != 0);

                const WordType eq = get_query_pattern(query_patterns, idx, pattern_idx_offset + t - t_begin + 1, target_begin[t - 1], false);
                if (idx == n_words_band - 1)
                {
                    // bits who have no left neighbor -> assume worst case: +1
                    pv_local |= carry_down_bit;
                    mv_local &= ~carry_down_bit;
                }

                const int32_t carry_right = myers_advance_block(warp_mask, carry_right_bit, eq, pv_local, mv_local, carry_down);
                carry_down                = ((pv_local & carry_down_bit) == WordType(0) ? 0 : 1) - ((mv_local & carry_down_bit) == WordType(0) ? 0 : 1);
                score(idx, t)             = score(idx, t - 1) + carry_right + carry_down;
                if (threadIdx.x == 0)
                    carry_down = 0;
                if (warp_mask == 0xffff'ffffu && (threadIdx.x == 0 || threadIdx.x == 31))
                    carry_down = __shfl_down_sync(0x8000'0001u, carry_down, warp_size - 1);
                if (threadIdx.x != 0)
                    carry_down = 0;
                pv(idx, t) = pv_local;
                mv(idx, t) = mv_local;
            }
            __syncwarp();
        }
    }
}

__device__ void
myers_compute_scores_edit_dist_banded(
    int32_t& diagonal_begin,
    int32_t& diagonal_end,
    device_matrix_view<WordType>& pv,
    device_matrix_view<WordType>& mv,
    device_matrix_view<int32_t>& score,
    device_matrix_view<WordType>& query_patterns,
    char const* target_begin,
    char const* query_begin,
    int32_t const target_size,
    int32_t const query_size,
    int32_t const band_width,
    int32_t const n_words_band,
    int32_t const p,
    int32_t const alignment_idx)
{
    // Note: 0-th row of the NW matrix is implicit for pv, mv and score! (given by the inital warp_carry)
    assert(warpSize == warp_size);
    assert(threadIdx.x < warp_size);
    assert(blockIdx.x == 0);

    assert(target_size > 0);
    assert(query_size > 0);

    assert(pv.num_rows() == n_words_band);
    assert(mv.num_rows() == n_words_band);
    assert(score.num_rows() == n_words_band);
    assert(pv.num_cols() == target_size + 1);
    assert(mv.num_cols() == target_size + 1);
    assert(score.num_cols() == target_size + 1);

    for (int32_t idx = threadIdx.x; idx < n_words_band; idx += warp_size)
    {
        pv(idx, 0)    = ~WordType(0);
        mv(idx, 0)    = 0;
        score(idx, 0) = min((idx + 1) * word_size, band_width);
    }
    __syncwarp();

    // This function computes a diagonal band of the NW matrix (Ukkonen algorithm).
    // In essence it computes the diagonals [-p, ..., 0, ..., p + target_size - query_size] (for query_size < target_size),
    // where diagonal -p starts at m(p,0), and p + target_size - query_size starts at m(0,p+target_size-query_size)
    // using Myers bit-vector algorithm with a word size of warp_size * sizeof(WordType).
    //
    // band_width is the width of this band = 1 + 2*p + abs(target_size - query_size).
    //
    // Note that for query_size >= target_size the diagonals [-p - (query_size - target_size), ..., 0, ..., p] are used.

    // This implementation computes the matrix band column by column.
    // To ease implementation band_width elements per column are computed for every column,
    // even though they are not needed for the first few and last few columns.
    //
    // In more detail: instead of just computing the diagonals:
    //
    // \\\\\00000|
    // \\\\\\0000|   target_size=9, query_size=7, p=1
    // 0\\\\\\000|
    // 00\\\\\\00|   ("|" has no meaning - just to avoid multi-line comments with trailing"\")
    // 000\\\\\\0|
    // 0000\\\\\\|
    // 00000\\\\\|
    //
    // we compute horizontal stripes with n=band_width rows at the beginning and at the end.
    // Only the range [diagonal_begin,diagonal_end)
    //
    // ----\00000|
    // ----\\0000|
    // ----\\----|
    // ----\\----|
    // ----\\----|
    // 0000\\----|
    // 00000\----|

    if (band_width >= query_size)
    {
        // If the band_width is larger than the query_size just do a full Myers
        // i.e. do only one large horizontal stripe of width query_size.
        diagonal_begin = target_size + 1;
        diagonal_end   = target_size + 1;
        myers_compute_scores_horizontal_band_impl(pv, mv, score, query_patterns, target_begin, query_begin, target_size, 1, target_size + 1, query_size, n_words_band, 0);
    }
    else
    {
        diagonal_begin = query_size < target_size ? target_size - query_size + p + 2 : p + 2;
        diagonal_end   = query_size < target_size ? query_size - p + 1 : query_size - (query_size - target_size) - p + 1;

        myers_compute_scores_horizontal_band_impl(pv, mv, score, query_patterns, target_begin, query_begin, target_size, 1, diagonal_begin, band_width, n_words_band, 0);
        myers_compute_scores_diagonal_band_impl(pv, mv, score, query_patterns, target_begin, query_begin, target_size, diagonal_begin, diagonal_end, band_width, n_words_band, 0);
        myers_compute_scores_horizontal_band_impl(pv, mv, score, query_patterns, target_begin, query_begin, target_size, diagonal_end, target_size + 1, band_width, n_words_band, query_size - band_width);
    }
}

__global__ void myers_banded_kernel(
    int8_t* paths_base,
    int32_t* path_lengths,
    const int32_t max_path_length,
    batched_device_matrices<WordType>::device_interface* pvi,
    batched_device_matrices<WordType>::device_interface* mvi,
    batched_device_matrices<int32_t>::device_interface* scorei,
    batched_device_matrices<WordType>::device_interface* query_patternsi,
    char const* sequences_d, int32_t const* sequence_lengths_d,
    int32_t max_sequence_length,
    int32_t n_alignments)
{
    assert(warpSize == warp_size);
    assert(threadIdx.x < warp_size);
    assert(blockIdx.x == 0);

    const int32_t alignment_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (alignment_idx >= n_alignments)
        return;
    const int32_t query_size  = sequence_lengths_d[2 * alignment_idx];
    const int32_t target_size = sequence_lengths_d[2 * alignment_idx + 1];
    const char* const query   = sequences_d + 2 * alignment_idx * max_sequence_length;
    const char* const target  = sequences_d + (2 * alignment_idx + 1) * max_sequence_length;
    const int32_t n_words     = (query_size + word_size - 1) / word_size;
    int8_t* path              = paths_base + alignment_idx * static_cast<ptrdiff_t>(max_path_length);

    device_matrix_view<WordType> query_pattern = query_patternsi->get_matrix_view(alignment_idx, n_words, 4);

    for (int32_t idx = threadIdx.x; idx < n_words; idx += warp_size)
    {
        // TODO query load is inefficient
        query_pattern(idx, 0) = myers_generate_query_pattern('A', query, query_size, idx * word_size);
        query_pattern(idx, 1) = myers_generate_query_pattern('C', query, query_size, idx * word_size);
        query_pattern(idx, 2) = myers_generate_query_pattern('T', query, query_size, idx * word_size);
        query_pattern(idx, 3) = myers_generate_query_pattern('G', query, query_size, idx * word_size);
    }
    __syncwarp();

    assert(query_size > 0);
    // Use the Ukkonen algorithm for banding.
    // Take an initial guess for the edit distance: max_distance_estimate
    // and compute the maximal band of the NW matrix which is required for this distance.
    // If the computed distance is smaller accept and compute the backtrace/path,
    // otherwise retry with a larger guess (i.e. and larger band).
    int32_t max_distance_estimate = max(1, abs(target_size - query_size) + min(target_size, query_size) / initial_distance_guess_factor);
    while (1)
    {
        int32_t p          = min3(target_size, query_size, (max_distance_estimate - abs(target_size - query_size)) / 2);
        int32_t band_width = min(1 + 2 * p + abs(target_size - query_size), query_size);
        if (band_width % word_size == 1 && band_width != query_size) // we need at least two bits in the last word
        {
            p += 1;
            band_width = min(1 + 2 * p + abs(target_size - query_size), query_size);
        }
        const int32_t n_words_band = ceiling_divide(band_width, word_size);

        device_matrix_view<WordType> pv   = pvi->get_matrix_view(alignment_idx, n_words_band, target_size + 1);
        device_matrix_view<WordType> mv   = mvi->get_matrix_view(alignment_idx, n_words_band, target_size + 1);
        device_matrix_view<int32_t> score = scorei->get_matrix_view(alignment_idx, n_words_band, target_size + 1);
        int32_t diagonal_begin            = -1;
        int32_t diagonal_end              = -1;
        myers_compute_scores_edit_dist_banded(diagonal_begin, diagonal_end, pv, mv, score, query_pattern, target, query, target_size, query_size, band_width, n_words_band, p, alignment_idx);
        __syncwarp();
        const int32_t cur_edit_distance = score(n_words_band - 1, target_size);
        if (cur_edit_distance <= max_distance_estimate || band_width == query_size)
        {
            if (threadIdx.x == 0)
            {
                const int32_t path_length   = myers_backtrace_banded(path, pv, mv, score, diagonal_begin, diagonal_end, band_width, target_size, query_size);
                path_lengths[alignment_idx] = path_length;
            }
            break;
        }
        max_distance_estimate *= 2;
    }
}

} // namespace myers

int32_t myers_compute_edit_distance(std::string const& target, std::string const& query)
{
    constexpr int32_t warp_size = 32;
    constexpr int32_t word_size = sizeof(myers::WordType) * CHAR_BIT;
    if (get_size(query) == 0)
        return get_size(target);

    const int32_t n_words = (get_size(query) + word_size - 1) / word_size;
    matrix<int32_t> score_host;

    cudaStream_t stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&stream));
    {
        DefaultDeviceAllocator allocator = create_default_device_allocator();

        int32_t max_sequence_length = std::max(get_size(target), get_size(query));
        device_buffer<char> sequences_d(2 * max_sequence_length, allocator, stream);
        device_buffer<int32_t> sequence_lengths_d(2, allocator, stream);

        batched_device_matrices<myers::WordType> pv(1, n_words * (get_size(target) + 1), allocator, stream);
        batched_device_matrices<myers::WordType> mv(1, n_words * (get_size(target) + 1), allocator, stream);
        batched_device_matrices<int32_t> score(1, n_words * (get_size(target) + 1), allocator, stream);
        batched_device_matrices<myers::WordType> query_patterns(1, n_words * 4, allocator, stream);

        std::array<int32_t, 2> lengths = {static_cast<int32_t>(get_size(query)), static_cast<int32_t>(get_size(target))};
        CGA_CU_CHECK_ERR(cudaMemcpyAsync(sequences_d.data(), query.data(), sizeof(char) * get_size(query), cudaMemcpyHostToDevice, stream));
        CGA_CU_CHECK_ERR(cudaMemcpyAsync(sequences_d.data() + max_sequence_length, target.data(), sizeof(char) * get_size(target), cudaMemcpyHostToDevice, stream));
        CGA_CU_CHECK_ERR(cudaMemcpyAsync(sequence_lengths_d.data(), lengths.data(), sizeof(int32_t) * 2, cudaMemcpyHostToDevice, stream));

        myers::myers_compute_score_matrix_kernel<<<1, warp_size, 0, stream>>>(pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), query_patterns.get_device_interface(), sequences_d.data(), sequence_lengths_d.data(), max_sequence_length, 1);

        score_host = score.get_matrix(0, n_words, get_size(target) + 1, stream);
        CGA_CU_CHECK_ERR(cudaStreamSynchronize(stream));
    }
    CGA_CU_CHECK_ERR(cudaStreamDestroy(stream));
    return score_host(n_words - 1, get_size(target));
}

matrix<int32_t> myers_get_full_score_matrix(std::string const& target, std::string const& query)
{
    constexpr int32_t warp_size = 32;
    constexpr int32_t word_size = sizeof(myers::WordType) * CHAR_BIT;

    if (get_size(target) == 0)
    {
        matrix<int32_t> r(get_size(query) + 1, 1);
        std::iota(r.data(), r.data() + get_size(query) + 1, 0);
        return r;
    }
    if (get_size(query) == 0)
    {
        matrix<int32_t> r(1, get_size(target) + 1);
        std::iota(r.data(), r.data() + get_size(target) + 1, 0);
        return r;
    }

    matrix<int32_t> fullscore_host;

    cudaStream_t stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&stream));

    {
        DefaultDeviceAllocator allocator = create_default_device_allocator();
        int32_t max_sequence_length      = std::max(get_size(target), get_size(query));
        device_buffer<char> sequences_d(2 * max_sequence_length, allocator, stream);
        device_buffer<int32_t> sequence_lengths_d(2, allocator, stream);

        const int32_t n_words = (get_size(query) + word_size - 1) / word_size;
        batched_device_matrices<myers::WordType> pv(1, n_words * (get_size(target) + 1), allocator, stream);
        batched_device_matrices<myers::WordType> mv(1, n_words * (get_size(target) + 1), allocator, stream);
        batched_device_matrices<int32_t> score(1, n_words * (get_size(target) + 1), allocator, stream);
        batched_device_matrices<myers::WordType> query_patterns(1, n_words * 4, allocator, stream);

        batched_device_matrices<int32_t> fullscore(1, (get_size(query) + 1) * (get_size(target) + 1), allocator, stream);

        std::array<int32_t, 2> lengths = {static_cast<int32_t>(get_size(query)), static_cast<int32_t>(get_size(target))};
        CGA_CU_CHECK_ERR(cudaMemcpyAsync(sequences_d.data(), query.data(), sizeof(char) * get_size(query), cudaMemcpyHostToDevice, stream));
        CGA_CU_CHECK_ERR(cudaMemcpyAsync(sequences_d.data() + max_sequence_length, target.data(), sizeof(char) * get_size(target), cudaMemcpyHostToDevice, stream));
        CGA_CU_CHECK_ERR(cudaMemcpyAsync(sequence_lengths_d.data(), lengths.data(), sizeof(int32_t) * 2, cudaMemcpyHostToDevice, stream));

        myers::myers_compute_score_matrix_kernel<<<1, warp_size, 0, stream>>>(pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), query_patterns.get_device_interface(), sequences_d.data(), sequence_lengths_d.data(), max_sequence_length, 1);
        {
            dim3 n_threads = {32, 4, 1};
            dim3 n_blocks  = {1, 1, 1};
            n_blocks.x     = ceiling_divide<int32_t>(get_size<int32_t>(query) + 1, n_threads.x);
            n_blocks.y     = ceiling_divide<int32_t>(get_size<int32_t>(target) + 1, n_threads.y);
            myers::myers_convert_to_full_score_matrix_kernel<<<n_blocks, n_threads, 0, stream>>>(fullscore.get_device_interface(), pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), sequence_lengths_d.data(), 0);
        }

        fullscore_host = fullscore.get_matrix(0, get_size(query) + 1, get_size(target) + 1, stream);
    }

    CGA_CU_CHECK_ERR(cudaStreamSynchronize(stream));
    CGA_CU_CHECK_ERR(cudaStreamDestroy(stream));
    return fullscore_host;
}

void myers_gpu(int8_t* paths_d, int32_t* path_lengths_d, int32_t max_path_length,
               char const* sequences_d,
               int32_t const* sequence_lengths_d,
               int32_t max_sequence_length,
               int32_t n_alignments,
               batched_device_matrices<myers::WordType>& pv,
               batched_device_matrices<myers::WordType>& mv,
               batched_device_matrices<int32_t>& score,
               batched_device_matrices<myers::WordType>& query_patterns,
               cudaStream_t stream)
{
    {
        const dim3 threads(warp_size, 1, 1);
        const dim3 blocks(1, ceiling_divide<int32_t>(n_alignments, threads.y), 1);
        myers::myers_compute_score_matrix_kernel<<<blocks, threads, 0, stream>>>(pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), query_patterns.get_device_interface(), sequences_d, sequence_lengths_d, max_sequence_length, n_alignments);
    }
    {
        const dim3 threads(128, 1, 1);
        const dim3 blocks(ceiling_divide<int32_t>(n_alignments, threads.x), 1, 1);
        myers::myers_backtrace_kernel<<<blocks, threads, 0, stream>>>(paths_d, path_lengths_d, max_path_length, pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), sequence_lengths_d, n_alignments);
    }
}

void myers_banded_gpu(int8_t* paths_d, int32_t* path_lengths_d, int32_t max_path_length,
                      char const* sequences_d,
                      int32_t const* sequence_lengths_d,
                      int32_t max_sequence_length,
                      int32_t n_alignments,
                      batched_device_matrices<myers::WordType>& pv,
                      batched_device_matrices<myers::WordType>& mv,
                      batched_device_matrices<int32_t>& score,
                      batched_device_matrices<myers::WordType>& query_patterns,
                      cudaStream_t stream)
{
    const dim3 threads(warp_size, 1, 1);
    const dim3 blocks(1, ceiling_divide<int32_t>(n_alignments, threads.y), 1);
    myers::myers_banded_kernel<<<blocks, threads, 0, stream>>>(paths_d, path_lengths_d, max_path_length, pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), query_patterns.get_device_interface(), sequences_d, sequence_lengths_d, max_sequence_length, n_alignments);
}

} // namespace cudaaligner
} // namespace claragenomics
