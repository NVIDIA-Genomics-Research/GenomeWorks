/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "cudapoa_kernels.cuh"
#include "cudapoa_batch.hpp"

#include <claragenomics/cudapoa/batch.hpp>

namespace claragenomics
{

namespace cudapoa
{

std::unique_ptr<Batch> create_batch(int32_t device_id,
                                    cudaStream_t stream,
                                    size_t max_mem,
                                    int8_t output_mask,
                                    const BatchSize& batch_size,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score,
                                    bool cuda_banded_alignment)
{
    if (use32bitScore(batch_size, gap_score, mismatch_score, match_score))
    {
        if (use32bitSize(batch_size, cuda_banded_alignment))
        {
            return std::make_unique<CudapoaBatch<int32_t, int32_t>>(device_id,
                                                                    stream,
                                                                    max_mem,
                                                                    output_mask,
                                                                    batch_size,
                                                                    (int32_t)gap_score,
                                                                    (int32_t)mismatch_score,
                                                                    (int32_t)match_score,
                                                                    cuda_banded_alignment);
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
                                                                    (int32_t)match_score,
                                                                    cuda_banded_alignment);
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
                                                                match_score,
                                                                cuda_banded_alignment);
    }
}

} // namespace cudapoa

} // namespace claragenomics
