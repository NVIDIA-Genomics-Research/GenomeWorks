/*
* Copyright 2019-2021 NVIDIA CORPORATION.
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

class AlignerGlobalMyersBanded : public FixedBandAligner
{
public:
    AlignerGlobalMyersBanded(int64_t max_device_memory, int32_t max_bandwidth, DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id);
    ~AlignerGlobalMyersBanded() override;

    StatusType align_all() override;
    StatusType sync_alignments() override;
    int32_t num_alignments() const override;

    void reset() override;
    void free_temporary_device_buffers() override;

    StatusType add_alignment(const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target) override;
    StatusType add_alignment(int32_t max_bandwidth, const char* query, int32_t query_length, const char* target, int32_t target_length, bool reverse_complement_query, bool reverse_complement_target) override;

    const std::vector<std::shared_ptr<Alignment>>& get_alignments() const override
    {
        return alignments_;
    }

    DeviceAlignmentsPtrs get_alignments_device() const override;

    cudaStream_t get_stream() const override
    {
        return stream_;
    }

    int32_t get_device() const override
    {
        return device_id_;
    }

    DefaultDeviceAllocator get_device_allocator() const override;

    void reset_max_bandwidth(int32_t max_bandwidth) override;

private:
    struct InternalData;

    static void reallocate_internal_data(InternalData* data, int64_t max_device_memory, int32_t max_bandwidth, int32_t n_alignments_initial, cudaStream_t stream);
    void reset_data();

    bool fits_device_memory(int64_t matrix_size_large, int64_t matrix_size_small, int32_t query_pattern_size_large, int32_t query_pattern_size_small, int32_t query_length, int32_t target_length) const;

    std::unique_ptr<InternalData> data_;
    cudaStream_t stream_;
    int32_t device_id_;
    int32_t max_bandwidth_;
    std::vector<std::shared_ptr<Alignment>> alignments_;
    int64_t max_device_memory_;
};

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
