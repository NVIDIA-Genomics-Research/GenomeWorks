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

std::unique_ptr<Batch> create_batch(int32_t device_id,
                                    cudaStream_t stream,
                                    size_t max_mem,
                                    int8_t output_mask,
                                    const BatchSize& batch_size,
                                    int16_t gap_score,
                                    int16_t mismatch_score,
                                    int16_t match_score,
                                    bool cuda_banded_alignment,
                                    bool cuda_adaptive_alignment)
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
                                                                    cuda_banded_alignment,
                                                                    cuda_adaptive_alignment);
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
                                                                    cuda_banded_alignment,
                                                                    cuda_adaptive_alignment);
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
                                                                cuda_banded_alignment,
                                                                cuda_adaptive_alignment);
    }
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
