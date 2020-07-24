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

#include "aligner_global.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

class AlignerGlobalMyersBanded : public Aligner
{
public:
    AlignerGlobalMyersBanded(int64_t max_device_memory, int32_t max_bandwidth, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id);
    ~AlignerGlobalMyersBanded() override;

    StatusType align_all() override;
    StatusType sync_alignments() override;
    void reset() override;

    StatusType add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target) override;

    const std::vector<std::shared_ptr<Alignment>>& get_alignments() const override
    {
        return alignments_;
    }

private:
    void reset_data();

    struct InternalData;
    std::unique_ptr<InternalData> data_;
    cudaStream_t stream_;
    int32_t device_id_;
    int32_t max_bandwidth_;
    std::vector<std::shared_ptr<Alignment>> alignments_;
};

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
