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

#include "myers_gpu.cuh"
#include "batched_device_matrices.cuh"

#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <claraparabricks/genomeworks/utils/mathutils.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/allocator.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>

#include <cassert>
#include <cuda/std/limits>
#include <vector>
#include <numeric>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <cuda/atomic>
#pragma GCC diagnostic pop

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

constexpr int32_t warp_size = 32;
constexpr int32_t word_size = sizeof(myers::WordType) * CHAR_BIT;

namespace myers
{

constexpr int32_t initial_distance_guess_factor = 20;

__global__ void init_atomic(cuda::atomic<int32_t, cuda::thread_scope_device>* atomic)
{
    // Safety-check for work-around for missing cuda::atomic_ref in libcu++ (see further below).
    static_assert(sizeof(int32_t) == sizeof(cuda::atomic<int32_t, cuda::thread_scope_device>), "cuda::atomic<int32_t> needs to have the same size as int32_t.");
    static_assert(alignof(int32_t) == alignof(cuda::atomic<int32_t, cuda::thread_scope_device>), "cuda::atomic<int32_t> needs to have the same alignment as int32_t.");
    atomic->store(0, cuda::memory_order_relaxed);
}

inline __device__ WordType warp_leftshift_sync(uint32_t warp_mask, WordType v)
{
    assert(((warp_mask >> (threadIdx.x % warp_size)) & 1u) == 1u);
    // 4 threads, word_size = 4 example: thread 0  | thread 1 | thread 2 | thread 3
    // v = 0101 | 0111 | 0011 | 1101 -> 1010 | 1110 | 0111 | 1010
    const WordType x = __shfl_up_sync(warp_mask, v >> (word_size - 1), 1);
    assert((x & ~WordType(1)) == 0);
    v <<= 1;
    if (threadIdx.x != 0)
        v |= x;
    return v;
}

inline __device__ WordType warp_rightshift_sync(uint32_t warp_mask, WordType v)
{
    assert(((warp_mask >> (threadIdx.x % warp_size)) & 1u) == 1u);
    // 4 threads, word_size = 4 example: thread 0  | thread 1 | thread 2 | thread 3
    // v = 0101 | 0111 | 0011 | 1101 -> 0010 | 1011 | 1001 | 1110
    const WordType x = __shfl_down_sync(warp_mask, v << (word_size - 1), 1);
    assert((x & ~(WordType(1) << (word_size - 1))) == 0);
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

__device__ int2 myers_advance_block2(uint32_t warp_mask, WordType highest_bit, WordType eq, WordType& pv, WordType& mv, int32_t carry_in)
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

    int2 carry_out;
    carry_out.x = ((ph & highest_bit) == WordType(0) ? 0 : 1) - ((mh & highest_bit) == WordType(0) ? 0 : 1);
    carry_out.y = ((ph & (highest_bit << 1)) == WordType(0) ? 0 : 1) - ((mh & (highest_bit << 1)) == WordType(0) ? 0 : 1);

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

    nw_score_t myscore = i > 0 ? score((i - 1) / word_size, j) : 0; // row 0 is implicit, NW matrix is shifted by i -> i-1 (see get_myers_score)
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
    GW_CONSTEXPR int32_t word_size          = sizeof(WordType) * CHAR_BIT;
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
    GW_CONSTEXPR int32_t word_size = sizeof(WordType) * CHAR_BIT;
    const int32_t query_size       = sequence_lengths_d[2 * alignment];
    const int32_t target_size      = sequence_lengths_d[2 * alignment + 1];
    const int32_t n_words          = (query_size + word_size - 1) / word_size;

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
    GW_CONSTEXPR int32_t word_size = sizeof(WordType) * CHAR_BIT;
    GW_CONSTEXPR int32_t warp_size = 32;
    assert(warpSize == warp_size);
    assert(threadIdx.x < warp_size);

    const int32_t alignment_idx = blockIdx.x;
    if (alignment_idx >= n_alignments)
        return;
    const int32_t query_size        = sequence_lengths_d[2 * alignment_idx];
    const int32_t target_size       = sequence_lengths_d[2 * alignment_idx + 1];
    const char* const query         = sequences_d + 2 * alignment_idx * max_sequence_length;
    const char* const target        = sequences_d + (2 * alignment_idx + 1) * max_sequence_length;
    const int32_t n_words           = (query_size + word_size - 1) / word_size;
    const int32_t n_warp_iterations = ceiling_divide(n_words, warp_size) * warp_size;

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

__device__ int32_t myers_backtrace_banded(int8_t* path, int32_t* const path_count, device_matrix_view<WordType> const& pv, device_matrix_view<WordType> const& mv, device_matrix_view<int32_t> const& score, int32_t diagonal_begin, int32_t diagonal_end, int32_t band_width, int32_t target_size, int32_t query_size)
{
    assert(threadIdx.x == 0);
    using nw_score_t                    = int32_t;
    GW_CONSTEXPR nw_score_t out_of_band = cuda::std::numeric_limits<nw_score_t>::max() - 1; // -1 to avoid integer overflow further down.
    assert(pv.num_rows() == score.num_rows());
    assert(mv.num_rows() == score.num_rows());
    assert(pv.num_cols() == score.num_cols());
    assert(mv.num_cols() == score.num_cols());
    assert(score.num_rows() == ceiling_divide(band_width, word_size));
    assert(diagonal_begin >= 0);
    assert(diagonal_end >= diagonal_begin);
    assert(diagonal_end >= 2); // this should only break if target_size == 0 - which is not valid input.
    assert(band_width > 0 || query_size == 0);

    int32_t i = band_width;
    int32_t j = target_size;

    const WordType last_entry_mask = band_width % word_size != 0 ? (WordType(1) << (band_width % word_size)) - 1 : ~WordType(0);

    const nw_score_t last_diagonal_score = diagonal_end < 2 ? out_of_band : get_myers_score(1, diagonal_end - 2, pv, mv, score, last_entry_mask) + 2;
    nw_score_t myscore                   = i > 0 ? score((i - 1) / word_size, j) : 0; // row 0 is implicit, NW matrix is shifted by i -> i-1, i.e. i \in [1,band_width] for get_myers_score. (see get_myers_score)
    int32_t pos                          = 0;
    int8_t prev_r                        = -1;
    int32_t r_count                      = 0;
    while (j >= diagonal_end)
    {
        int8_t r = 0;
        // Worst case for the implicit top row (i == 0) of the bottom right block of the NW is the last diagonal entry on the same row + (j - diagonal_end) * indel cost.
        nw_score_t const above = i <= 1 ? (last_diagonal_score + j - diagonal_end) : get_myers_score(i - 1, j, pv, mv, score, last_entry_mask);
        nw_score_t const diag  = i <= 1 ? (last_diagonal_score + j - 1 - diagonal_end) : get_myers_score(i - 1, j - 1, pv, mv, score, last_entry_mask);
        nw_score_t const left  = i < 1 ? (last_diagonal_score + j - 1 - diagonal_end) : get_myers_score(i, j - 1, pv, mv, score, last_entry_mask);
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
            assert(diag == myscore || diag + 1 == myscore);
            r       = (diag == myscore ? static_cast<int8_t>(AlignmentState::match) : static_cast<int8_t>(AlignmentState::mismatch));
            myscore = diag;
            --i;
            --j;
        }
        if (prev_r != r)
        {
            if (prev_r != -1)
            {
                path[pos]       = prev_r;
                path_count[pos] = r_count;
                ++pos;
            }
            prev_r  = r;
            r_count = 0;
        }
        ++r_count;
    }
    while (j >= diagonal_begin)
    {
        int8_t r               = 0;
        nw_score_t const above = i <= 1 ? out_of_band : get_myers_score(i - 1, j, pv, mv, score, last_entry_mask);
        nw_score_t const diag  = i <= 0 ? j - 1 : get_myers_score(i, j - 1, pv, mv, score, last_entry_mask);
        nw_score_t const left  = i >= band_width ? out_of_band : get_myers_score(i + 1, j - 1, pv, mv, score, last_entry_mask);
        // out-of-band cases: diag always preferrable, since worst-case-(above|left) - myscore >= diag - myscore always holds.
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
            assert(diag == myscore || diag + 1 == myscore);
            r       = (diag == myscore ? static_cast<int8_t>(AlignmentState::match) : static_cast<int8_t>(AlignmentState::mismatch));
            myscore = diag;
            --j;
        }
        if (prev_r != r)
        {
            if (prev_r != -1)
            {
                path[pos]       = prev_r;
                path_count[pos] = r_count;
                ++pos;
            }
            prev_r  = r;
            r_count = 0;
        }
        ++r_count;
    }
    while (i > 0 && j > 0)
    {
        int8_t r               = 0;
        nw_score_t const above = i == 1 ? j : get_myers_score(i - 1, j, pv, mv, score, last_entry_mask);
        nw_score_t const diag  = i == 1 ? j - 1 : get_myers_score(i - 1, j - 1, pv, mv, score, last_entry_mask);
        nw_score_t const left  = i > band_width ? out_of_band : get_myers_score(i, j - 1, pv, mv, score, last_entry_mask);
        // out-of-band cases: diag always preferrable, since worst-case-(above|left) - myscore >= diag - myscore always holds.
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
            assert(diag == myscore || diag + 1 == myscore);
            r       = (diag == myscore ? static_cast<int8_t>(AlignmentState::match) : static_cast<int8_t>(AlignmentState::mismatch));
            myscore = diag;
            --i;
            --j;
        }
        if (prev_r != r)
        {
            if (prev_r != -1)
            {
                path[pos]       = prev_r;
                path_count[pos] = r_count;
                ++pos;
            }
            prev_r  = r;
            r_count = 0;
        }
        ++r_count;
    }
    if (i > 0)
    {
        if (prev_r != static_cast<int8_t>(AlignmentState::deletion))
        {
            if (prev_r != -1)
            {
                path[pos]       = prev_r;
                path_count[pos] = r_count;
                ++pos;
            }
            prev_r  = static_cast<int8_t>(AlignmentState::deletion);
            r_count = 0;
        }
        r_count += i;
    }
    if (j > 0)
    {
        if (prev_r != static_cast<int8_t>(AlignmentState::insertion))
        {
            if (prev_r != -1)
            {
                path[pos]       = prev_r;
                path_count[pos] = r_count;
                ++pos;
            }
            prev_r  = static_cast<int8_t>(AlignmentState::insertion);
            r_count = 0;
        }
        r_count += j;
    }
    if (r_count != 0)
    {
        assert(prev_r != -1);
        path[pos]       = prev_r;
        path_count[pos] = r_count;
        ++pos;
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
    const int32_t t_begin,
    const int32_t t_end,
    const int32_t width,
    const int32_t n_words,
    const int32_t pattern_idx_offset)
{
    assert(n_words == ceiling_divide(width, word_size));
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
        int32_t carry = 0;
        if (threadIdx.x == 0)
            carry = 1; // worst case for the top boarder of the band
        for (int32_t idx = threadIdx.x; idx < n_warp_iterations; idx += warp_size)
        {
            // idx within band column
            const uint32_t warp_mask = idx / warp_size < n_words_band / warp_size ? 0xffff'ffffu : (1u << (n_words_band % warp_size)) - 1;

            if (idx < n_words_band)
            {
                // data from the previous column
                WordType pv_local = warp_rightshift_sync(warp_mask, pv(idx, t - 1));
                WordType mv_local = warp_rightshift_sync(warp_mask, mv(idx, t - 1));
                if (threadIdx.x == 31 && warp_mask == 0xffff'ffffu)
                {
                    if (idx < n_words_band - 1)
                    {
                        pv_local |= pv(idx + 1, t - 1) << (word_size - 1);
                        mv_local |= mv(idx + 1, t - 1) << (word_size - 1);
                    }
                }

                const WordType eq = get_query_pattern(query_patterns, idx, pattern_idx_offset + t - t_begin + 1, target_begin[t - 1], false);

                const WordType delta_right_bit = WordType(1) << (idx == (n_words_band - 1) ? band_width - (n_words_band - 1) * word_size - 2 : word_size - 2);
                const WordType delta_down_bit  = delta_right_bit << 1;
                assert(delta_down_bit != 0);
                if (idx == n_words_band - 1)
                {
                    // bits who have no left neighbor -> assume worst case: +1
                    pv_local |= delta_down_bit;
                    mv_local &= ~delta_down_bit;
                }

                const int2 delta_right   = myers_advance_block2(warp_mask, delta_right_bit, eq, pv_local, mv_local, carry);
                const int32_t delta_down = ((pv_local & delta_down_bit) == WordType(0) ? 0 : 1) - ((mv_local & delta_down_bit) == WordType(0) ? 0 : 1);
                // Since idx is relative to diagonal band, (idx, t-1) -> (idx,t)
                // corresponds to (n-1,t-1) -> (n,t) in the NW matrix.
                // To get from score'(n-1, t-1) -> score'(n, t-1)
                // add horizontal delta in row n-1 (delta_right.x)
                // and the vertical delta in column t (delta_down).
                score(idx, t) = score(idx, t - 1) + delta_right.x + delta_down;

                // Carry horizontal delta in row n (= delta_right.y) to next warp iteration
                if (threadIdx.x == 0)
                    carry = 0;
                if (warp_mask == 0xffff'ffffu && (threadIdx.x == 0 || threadIdx.x == 31))
                    carry = __shfl_down_sync(0x8000'0001u, delta_right.y, warp_size - 1);
                if (threadIdx.x != 0)
                    carry = 0;

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

    assert(target_size > 0);
    assert(query_size >= 0);
    assert(band_width > 0 || query_size == 0); // might even be ok for band_width = 0 - haven't checked.
    assert(n_words_band > 0 || query_size == 0);
    assert(p >= 0);
    assert(alignment_idx >= 0);

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
        myers_compute_scores_horizontal_band_impl(pv, mv, score, query_patterns, target_begin, query_begin, 1, target_size + 1, query_size, n_words_band, 0);
    }
    else
    {
        const int32_t symmetric_band = (band_width - min(1 + 2 * p + abs(target_size - query_size), query_size) == 0) ? 1 : 0;
        diagonal_begin               = query_size < target_size ? target_size - query_size + p + 2 : p + 2 + (1 - symmetric_band);
        diagonal_end                 = query_size < target_size ? query_size - p + symmetric_band : query_size - (query_size - target_size) - p + 1;

        myers_compute_scores_horizontal_band_impl(pv, mv, score, query_patterns, target_begin, query_begin, 1, diagonal_begin, band_width, n_words_band, 0);
        myers_compute_scores_diagonal_band_impl(pv, mv, score, query_patterns, target_begin, query_begin, diagonal_begin, diagonal_end, band_width, n_words_band, 0);
        myers_compute_scores_horizontal_band_impl(pv, mv, score, query_patterns, target_begin, query_begin, diagonal_end, target_size + 1, band_width, n_words_band, query_size - band_width);
    }
}

__device__ int32_t get_alignment_task(const int32_t* scheduling_index_d, cuda::atomic<int32_t, cuda::thread_scope_device>* scheduling_atomic_d)
{
    // Fetch the index of the next alignment to be processed.
    // A full warp operates on the same alignment, i.e.
    // the whole warp gets the same alignment index.
    int32_t sched_idx = 0;
    if (threadIdx.x == 0)
    {
        sched_idx = scheduling_atomic_d->fetch_add(1, cuda::memory_order_relaxed);
    }
    sched_idx = __shfl_sync(0xffff'ffffu, sched_idx, 0);
    return scheduling_index_d[sched_idx];
}

__global__ void myers_banded_kernel(
    int8_t* paths_base,
    int32_t* const path_counts_base,
    int32_t* path_lengths,
    int64_t const* path_starts,
    batched_device_matrices<WordType>::device_interface* pvi,
    batched_device_matrices<WordType>::device_interface* mvi,
    batched_device_matrices<int32_t>::device_interface* scorei,
    batched_device_matrices<WordType>::device_interface* query_patternsi,
    char const* sequences_d, int64_t const* sequence_starts_d, int32_t const* max_bandwidths_d,
    const int32_t* scheduling_index_d, cuda::atomic<int32_t, cuda::thread_scope_device>* scheduling_atomic_d,
    const int32_t n_alignments)
{
    assert(warpSize == warp_size);
    assert(threadIdx.x < warp_size);

    if (blockIdx.x >= n_alignments)
        return;
    const int32_t alignment_idx = get_alignment_task(scheduling_index_d, scheduling_atomic_d);
    assert(alignment_idx < n_alignments);
    if (alignment_idx >= n_alignments)
        return;
    const char* const query     = sequences_d + sequence_starts_d[2 * alignment_idx];
    const char* const target    = sequences_d + sequence_starts_d[2 * alignment_idx + 1];
    const int32_t query_size    = target - query;
    const int32_t target_size   = sequences_d + sequence_starts_d[2 * alignment_idx + 2] - target;
    const int32_t n_words       = ceiling_divide(query_size, word_size);
    const int32_t max_bandwidth = max_bandwidths_d[alignment_idx];
    assert(max_bandwidth % word_size != 1); // we need at least two bits in the last word
    if (max_bandwidth - 1 < abs(target_size - query_size) && query_size != 0 && target_size != 0)
    {
        if (threadIdx.x == 0)
        {
            path_lengths[alignment_idx] = 0;
        }
        return;
    }
    if (target_size == 0 || query_size == 0)
    {
        // Temporary fix for edge cases target_size == 0 and query_size == 0.
        // TODO: check if the regular implementation works for this case.
        if (threadIdx.x == 0)
        {
            int8_t* const path         = paths_base + path_starts[alignment_idx];
            int32_t* const path_counts = path_counts_base + path_starts[alignment_idx];
            if (query_size == 0 && target_size == 0)
            {
                path_lengths[alignment_idx] = 0;
            }
            else
            {
                path[0]                     = query_size == 0 ? static_cast<int8_t>(AlignmentState::insertion) : static_cast<int8_t>(AlignmentState::deletion);
                path_counts[0]              = query_size + target_size; // one of them is 0.
                path_lengths[alignment_idx] = 1;
            }
        }
        return;
    }
    __syncthreads();

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

    // Use the Ukkonen algorithm for banding.
    // Take an initial guess for the edit distance: max_distance_estimate
    // and compute the maximal band of the NW matrix which is required for this distance.
    // If the computed distance is smaller accept and compute the backtrace/path,
    // otherwise retry with a larger guess (i.e. and larger band).
    int32_t max_distance_estimate = max(1, abs(target_size - query_size) + min(target_size, query_size) / initial_distance_guess_factor);
    device_matrix_view<WordType> pv;
    device_matrix_view<WordType> mv;
    device_matrix_view<int32_t> score;
    int32_t diagonal_begin = -1;
    int32_t diagonal_end   = -1;
    int32_t band_width     = 0;
    while (1)
    {
        int32_t p              = min3(target_size, query_size, (max_distance_estimate - abs(target_size - query_size)) / 2);
        int32_t band_width_new = min(1 + 2 * p + abs(target_size - query_size), query_size);
        if (band_width_new % word_size == 1 && band_width_new != query_size) // we need at least two bits in the last word
        {
            p += 1;
            band_width_new = min(1 + 2 * p + abs(target_size - query_size), query_size);
        }
        if (band_width_new > max_bandwidth)
        {
            band_width_new = max_bandwidth;
            p              = (band_width_new - 1 - abs(target_size - query_size)) / 2;
        }
        const int32_t n_words_band = ceiling_divide(band_width_new, word_size);
        if (static_cast<int64_t>(n_words_band) * static_cast<int64_t>(target_size + 1) > pvi->get_max_elements_per_matrix(alignment_idx))
        {
            band_width = -band_width;
            break;
        }
        band_width     = band_width_new;
        pv             = pvi->get_matrix_view(alignment_idx, n_words_band, target_size + 1);
        mv             = mvi->get_matrix_view(alignment_idx, n_words_band, target_size + 1);
        score          = scorei->get_matrix_view(alignment_idx, n_words_band, target_size + 1);
        diagonal_begin = -1;
        diagonal_end   = -1;
        myers_compute_scores_edit_dist_banded(diagonal_begin, diagonal_end, pv, mv, score, query_pattern, target, query, target_size, query_size, band_width, n_words_band, p, alignment_idx);
        __syncwarp();
        assert(n_words_band > 0 || query_size == 0);
        const int32_t cur_edit_distance = n_words_band > 0 ? score(n_words_band - 1, target_size) : target_size;
        if (cur_edit_distance <= max_distance_estimate || band_width == query_size)
        {
            break;
        }
        if (band_width == max_bandwidth)
        {
            band_width = -band_width;
            break;
        }
        max_distance_estimate *= 2;
    }
    if (threadIdx.x == 0)
    {
        int32_t path_length = 0;
        if (band_width != 0)
        {
            int8_t* const path         = paths_base + path_starts[alignment_idx];
            int32_t* const path_counts = path_counts_base + path_starts[alignment_idx];
            path_length                = band_width > 0 ? 1 : -1;
            band_width                 = abs(band_width);
            path_length *= myers_backtrace_banded(path, path_counts, pv, mv, score, diagonal_begin, diagonal_end, band_width, target_size, query_size);
        }
        path_lengths[alignment_idx] = path_length;
    }
}

} // namespace myers

int32_t myers_compute_edit_distance(std::string const& target, std::string const& query)
{
    if (get_size(query) == 0)
        return get_size(target);

    const int32_t n_words = (get_size(query) + word_size - 1) / word_size;

    CudaStream stream                = make_cuda_stream();
    DefaultDeviceAllocator allocator = create_default_device_allocator();

    int32_t max_sequence_length = std::max(get_size(target), get_size(query));
    device_buffer<char> sequences_d(2 * max_sequence_length, allocator, stream.get());
    device_buffer<int32_t> sequence_lengths_d(2, allocator, stream.get());

    batched_device_matrices<myers::WordType> pv(1, n_words * (get_size(target) + 1), allocator, stream.get());
    batched_device_matrices<myers::WordType> mv(1, n_words * (get_size(target) + 1), allocator, stream.get());
    batched_device_matrices<int32_t> score(1, n_words * (get_size(target) + 1), allocator, stream.get());
    batched_device_matrices<myers::WordType> query_patterns(1, n_words * 4, allocator, stream.get());

    std::array<int32_t, 2> lengths = {static_cast<int32_t>(get_size(query)), static_cast<int32_t>(get_size(target))};
    cudautils::device_copy_n_async(query.data(), get_size(query), sequences_d.data(), stream.get());
    cudautils::device_copy_n_async(target.data(), get_size(target), sequences_d.data() + max_sequence_length, stream.get());
    cudautils::device_copy_n_async(lengths.data(), 2, sequence_lengths_d.data(), stream.get());

    myers::myers_compute_score_matrix_kernel<<<1, warp_size, 0, stream.get()>>>(pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), query_patterns.get_device_interface(), sequences_d.data(), sequence_lengths_d.data(), max_sequence_length, 1);
    GW_CU_CHECK_ERR(cudaPeekAtLastError());

    matrix<int32_t> score_host = score.get_matrix(0, n_words, get_size(target) + 1, stream.get());
    return score_host(n_words - 1, get_size(target));
}

matrix<int32_t> myers_get_full_score_matrix(std::string const& target, std::string const& query)
{
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

    CudaStream stream = make_cuda_stream();

    DefaultDeviceAllocator allocator = create_default_device_allocator();
    int32_t max_sequence_length      = std::max(get_size(target), get_size(query));
    device_buffer<char> sequences_d(2 * max_sequence_length, allocator, stream.get());
    device_buffer<int32_t> sequence_lengths_d(2, allocator, stream.get());

    const int32_t n_words = (get_size(query) + word_size - 1) / word_size;
    batched_device_matrices<myers::WordType> pv(1, n_words * (get_size(target) + 1), allocator, stream.get());
    batched_device_matrices<myers::WordType> mv(1, n_words * (get_size(target) + 1), allocator, stream.get());
    batched_device_matrices<int32_t> score(1, n_words * (get_size(target) + 1), allocator, stream.get());
    batched_device_matrices<myers::WordType> query_patterns(1, n_words * 4, allocator, stream.get());

    batched_device_matrices<int32_t> fullscore(1, (get_size(query) + 1) * (get_size(target) + 1), allocator, stream.get());

    std::array<int32_t, 2> lengths = {static_cast<int32_t>(get_size(query)), static_cast<int32_t>(get_size(target))};
    cudautils::device_copy_n_async(query.data(), get_size(query), sequences_d.data(), stream.get());
    cudautils::device_copy_n_async(target.data(), get_size(target), sequences_d.data() + max_sequence_length, stream.get());
    cudautils::device_copy_n_async(lengths.data(), 2, sequence_lengths_d.data(), stream.get());

    myers::myers_compute_score_matrix_kernel<<<1, warp_size, 0, stream.get()>>>(pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), query_patterns.get_device_interface(), sequences_d.data(), sequence_lengths_d.data(), max_sequence_length, 1);
    GW_CU_CHECK_ERR(cudaPeekAtLastError());
    {
        dim3 n_threads = {32, 4, 1};
        dim3 n_blocks  = {1, 1, 1};
        n_blocks.x     = ceiling_divide<int32_t>(get_size<int32_t>(query) + 1, n_threads.x);
        n_blocks.y     = ceiling_divide<int32_t>(get_size<int32_t>(target) + 1, n_threads.y);
        myers::myers_convert_to_full_score_matrix_kernel<<<n_blocks, n_threads, 0, stream.get()>>>(fullscore.get_device_interface(), pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), sequence_lengths_d.data(), 0);
        GW_CU_CHECK_ERR(cudaPeekAtLastError());
    }

    matrix<int32_t> fullscore_host = fullscore.get_matrix(0, get_size(query) + 1, get_size(target) + 1, stream.get());
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
        const dim3 blocks(n_alignments, 1, 1);
        myers::myers_compute_score_matrix_kernel<<<blocks, threads, 0, stream>>>(pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), query_patterns.get_device_interface(), sequences_d, sequence_lengths_d, max_sequence_length, n_alignments);
    }
    {
        const dim3 threads(128, 1, 1);
        const dim3 blocks(ceiling_divide<int32_t>(n_alignments, threads.x), 1, 1);
        myers::myers_backtrace_kernel<<<blocks, threads, 0, stream>>>(paths_d, path_lengths_d, max_path_length, pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), sequence_lengths_d, n_alignments);
    }
    GW_CU_CHECK_ERR(cudaPeekAtLastError());
}

void myers_banded_gpu(int8_t* paths_d, int32_t* path_counts_d, int32_t* path_lengths_d, int64_t const* path_starts_d,
                      char const* sequences_d,
                      int64_t const* sequence_starts_d,
                      int32_t const* max_bandwidths_d,
                      int32_t const* scheduling_index_d,
                      int32_t* scheduling_atomic_int_d,
                      int32_t n_alignments,
                      batched_device_matrices<myers::WordType>& pv,
                      batched_device_matrices<myers::WordType>& mv,
                      batched_device_matrices<int32_t>& score,
                      batched_device_matrices<myers::WordType>& query_patterns,
                      cudaStream_t stream)
{
    const dim3 threads(warp_size, 1, 1);
    const dim3 blocks(n_alignments, 1, 1);

    // Work-around for missing cuda::atomic_ref in libcu++.
    static_assert(sizeof(int32_t) == sizeof(cuda::atomic<int32_t, cuda::thread_scope_device>), "cuda::atomic<int32_t> needs to have the same size as int32_t.");
    static_assert(alignof(int32_t) == alignof(cuda::atomic<int32_t, cuda::thread_scope_device>), "cuda::atomic<int32_t> needs to have the same alignment as int32_t.");
    cuda::atomic<int32_t, cuda::thread_scope_device>* const scheduling_atomic_d = reinterpret_cast<cuda::atomic<int32_t, cuda::thread_scope_device>*>(scheduling_atomic_int_d);

    myers::init_atomic<<<1, 1, 0, stream>>>(scheduling_atomic_d);
    myers::myers_banded_kernel<<<blocks, threads, 0, stream>>>(paths_d, path_counts_d, path_lengths_d, path_starts_d,
                                                               pv.get_device_interface(), mv.get_device_interface(), score.get_device_interface(), query_patterns.get_device_interface(),
                                                               sequences_d, sequence_starts_d, max_bandwidths_d, scheduling_index_d, scheduling_atomic_d, n_alignments);
    GW_CU_CHECK_ERR(cudaPeekAtLastError());
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
