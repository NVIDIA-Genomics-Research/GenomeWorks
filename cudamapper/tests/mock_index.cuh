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

#include "gmock/gmock.h"

#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/allocator.hpp>
#include "../src/index_gpu.cuh"
#include "../src/minimizer.hpp"
#include "cudamapper_file_location.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

class MockIndex : public IndexGPU<Minimizer>
{
public:
    MockIndex(DefaultDeviceAllocator allocator)
        : IndexGPU(allocator,
                   *genomeworks::io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta"),
                   0,
                   0,
                   0,
                   0,
                   true)
    {
    }

    MOCK_METHOD(device_buffer<read_id_t>&, read_ids, (), (const, override));
    MOCK_METHOD(device_buffer<position_in_read_t>&, positions_in_reads, (), (const, override));
    MOCK_METHOD(device_buffer<std::uint32_t>&, first_occurrence_of_representations, (), (const, override));
    MOCK_METHOD(read_id_t, number_of_reads, (), (const, override));
    MOCK_METHOD(read_id_t, smallest_read_id, (), (const, override));
    MOCK_METHOD(read_id_t, largest_read_id, (), (const, override));
    MOCK_METHOD(position_in_read_t, number_of_basepairs_in_longest_read, (), (const, override));
};

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
