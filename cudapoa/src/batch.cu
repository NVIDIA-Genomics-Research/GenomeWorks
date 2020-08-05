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

#include <memory>

#include "cudapoa_limits.hpp"
#include "cudapoa_batch.cuh"

#include <claraparabricks/genomeworks/cudapoa/batch.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

/// constructor- set other parameters based on a minimum set of input arguments
BatchConfig::BatchConfig(int32_t max_seq_sz /*= 1024*/, int32_t max_seq_per_poa /*= 100*/, int32_t band_width /*= 256*/, BandMode banding /*= BandMode::full_band*/)
    /// ensure a 4-byte boundary alignment for any allocated buffer
    : max_sequence_size(max_seq_sz)
    , max_consensus_size(2 * max_sequence_size)
    /// ensure 128-alignment for band_width size, 128 = CUDAPOA_MIN_BAND_WIDTH
    , alignment_band_width(cudautils::align<int32_t, CUDAPOA_MIN_BAND_WIDTH>(band_width))
    , max_sequences_per_poa(max_seq_per_poa)
    , band_mode(banding)
{
    if (banding == BandMode::full_band)
    {
        max_nodes_per_graph       = cudautils::align<int32_t, CELLS_PER_THREAD>(3 * max_sequence_size);
        matrix_graph_dimension    = cudautils::align<int32_t, CELLS_PER_THREAD>(max_nodes_per_graph);
        matrix_sequence_dimension = cudautils::align<int32_t, CELLS_PER_THREAD>(max_sequence_size);
    }
    else if (banding == BandMode::static_band)
    {
        max_nodes_per_graph    = cudautils::align<int32_t, CELLS_PER_THREAD>(4 * max_sequence_size);
        matrix_graph_dimension = cudautils::align<int32_t, CELLS_PER_THREAD>(max_nodes_per_graph);
        // 8 = CUDAPOA_BANDED_MATRIX_RIGHT_PADDING
        matrix_sequence_dimension = cudautils::align<int32_t, CELLS_PER_THREAD>(alignment_band_width + 8);
    }
    else // BandMode::adaptive_band
    {
        max_nodes_per_graph    = cudautils::align<int32_t, CELLS_PER_THREAD>(4 * max_sequence_size);
        matrix_graph_dimension = cudautils::align<int32_t, CELLS_PER_THREAD>(max_nodes_per_graph);
        // 8 = CUDAPOA_BANDED_MATRIX_RIGHT_PADDING, *2 is to reserve extra memory for cases with extended band-width
        matrix_sequence_dimension = cudautils::align<int32_t, CELLS_PER_THREAD>(2 * (alignment_band_width + 8));
    }

    throw_on_negative(max_seq_sz, "max_sequence_size cannot be negative.");
    throw_on_negative(max_seq_per_poa, "max_sequences_per_poa cannot be negative.");
    throw_on_negative(band_width, "alignment_band_width cannot be negative.");
    if (alignment_band_width != band_width)
    {
        std::cerr << "Band-width should be multiple of 128. The input was changed from " << band_width << " to " << alignment_band_width << std::endl;
    }
}

/// constructor- set all parameters separately
BatchConfig::BatchConfig(int32_t max_seq_sz, int32_t max_consensus_sz, int32_t max_nodes_per_w,
                         int32_t band_width, int32_t max_seq_per_poa, int32_t matrix_seq_dim, BandMode banding)
    /// ensure a 4-byte boundary alignment for any allocated buffer
    : max_sequence_size(max_seq_sz)
    , max_consensus_size(max_consensus_sz)
    , max_nodes_per_graph(cudautils::align<int32_t, CELLS_PER_THREAD>(max_nodes_per_w))
    , matrix_graph_dimension(cudautils::align<int32_t, CELLS_PER_THREAD>(max_nodes_per_graph))
    , matrix_sequence_dimension(cudautils::align<int32_t, CELLS_PER_THREAD>(matrix_seq_dim))
    /// ensure 128-alignment for band_width size
    , alignment_band_width(cudautils::align<int32_t, CUDAPOA_MIN_BAND_WIDTH>(band_width))
    , max_sequences_per_poa(max_seq_per_poa)
    , band_mode(banding)
{
    throw_on_negative(max_seq_sz, "max_sequence_size cannot be negative.");
    throw_on_negative(max_consensus_sz, "max_consensus_size cannot be negative.");
    throw_on_negative(max_nodes_per_w, "max_nodes_per_graph cannot be negative.");
    throw_on_negative(max_seq_per_poa, "max_sequences_per_poa cannot be negative.");
    throw_on_negative(band_width, "alignment_band_width cannot be negative.");

    if (max_nodes_per_graph < max_sequence_size)
        throw std::invalid_argument("max_nodes_per_graph should be greater than or equal to max_sequence_size.");
    if (max_consensus_size < max_sequence_size)
        throw std::invalid_argument("max_consensus_size should be greater than or equal to max_sequence_size.");
    if (max_sequence_size < alignment_band_width)
        throw std::invalid_argument("alignment_band_width should not be greater than max_sequence_size.");
    if (alignment_band_width != band_width)
    {
        std::cerr << "Band-width should be multiple of 128. The input was changed from " << band_width << " to " << alignment_band_width << std::endl;
    }
}

std::unique_ptr<Batch> create_batch(int32_t device_id,
                                    cudaStream_t stream,
                                    size_t max_mem,
                                    int8_t output_mask,
                                    const BatchConfig& batch_size,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score)
{
    if (use32bitScore(batch_size, gap_score, mismatch_score, match_score))
    {
        if (use32bitSize(batch_size))
        {
            return std::make_unique<CudapoaBatch<int32_t, int32_t>>(device_id,
                                                                    stream,
                                                                    max_mem,
                                                                    output_mask,
                                                                    batch_size,
                                                                    (int32_t)gap_score,
                                                                    (int32_t)mismatch_score,
                                                                    (int32_t)match_score);
        }
        else
        {
            return std::make_unique<CudapoaBatch<int32_t, int16_t>>(device_id,
                                                                    stream,
                                                                    max_mem,
                                                                    output_mask,
                                                                    batch_size,
                                                                    (int32_t)gap_score,
                                                                    (int32_t)mismatch_score,
                                                                    (int32_t)match_score);
        }
    }
    else
    {
        // if ScoreT is 16-bit, then it's safe to assume SizeT is 16-bit
        return std::make_unique<CudapoaBatch<int16_t, int16_t>>(device_id,
                                                                stream,
                                                                max_mem,
                                                                output_mask,
                                                                batch_size,
                                                                gap_score,
                                                                mismatch_score,
                                                                match_score);
    }
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
