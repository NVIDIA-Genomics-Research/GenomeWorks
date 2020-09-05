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

#include <claraparabricks/genomeworks/cudaextender/extender.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaextender
{

class UngappedXDrop : public Extender
{
public:
    UngappedXDrop(int32_t* h_sub_mat, bool no_entropy = false, cudaStream_t stream = 0, int32_t device_id = 0);
    ~UngappedXDrop() override;

    StatusType extend_async(const char* h_query, int32_t query_length,
                                    const char* h_target, int32_t target_length,
                                    int32_t score_threshold,
                                    std::vector<SeedPair>& h_seed_pairs) override;

    StatusType extend_async(const char* d_query, int32_t query_length,
                            const char* d_target, int32_t target_length,
                            int32_t score_threshold, SeedPair* d_seed_pairs,
                            int32_t num_seed_pairs, ScoredSegmentPair* d_scored_segment_pairs,
                            int32_t* d_num_scored_segment_pairs) override;

    StatusType sync() override;
    void reset() override;


    const std::vector<ScoredSegmentPair>& get_scored_segment_pairs() const override;

private:

    int32_t* h_sub_mat_;
    cudaStream_t stream_;
    int32_t device_id_;
    bool no_entropy_;
    std::vector<ScoredSegmentPair> scored_segment_pairs_;
};

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
