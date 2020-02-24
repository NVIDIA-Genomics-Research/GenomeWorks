/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "gmock/gmock.h"

#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/allocator.hpp>
#include "../src/index_gpu.cuh"
#include "../src/minimizer.hpp"
#include "cudamapper_file_location.hpp"

namespace claragenomics
{
namespace cudamapper
{

class MockIndex : public IndexGPU<Minimizer>
{
public:
    MockIndex(std::shared_ptr<DeviceAllocator> allocator)
        : IndexGPU(allocator,
                   *claragenomics::io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta"),
                   0,
                   0,
                   0,
                   0,
                   true)
    {
    }

    MOCK_METHOD(device_buffer<read_id_t>&, read_ids, (), (const, override));
    MOCK_METHOD(device_buffer<position_in_read_t>&, positions_in_reads, (), (const, override));
    MOCK_METHOD(const std::string&, read_id_to_read_name, (const read_id_t read_id), (const, override));
    MOCK_METHOD(device_buffer<std::uint32_t>&, first_occurrence_of_representations, (), (const, override));
    MOCK_METHOD(const std::uint32_t&, read_id_to_read_length, (const read_id_t read_id), (const, override));
    MOCK_METHOD(read_id_t, number_of_reads, (), (const, override));
    MOCK_METHOD(read_id_t, smallest_read_id, (), (const, override));
    MOCK_METHOD(read_id_t, largest_read_id, (), (const, override));
    MOCK_METHOD(position_in_read_t, number_of_basepairs_in_longest_read, (), (const, override));
};

} // namespace cudamapper
} // namespace claragenomics
