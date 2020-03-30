/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include "../src/index_cache.cuh"

namespace claragenomics
{
namespace cudamapper
{

TEST(TestCudamapperIndexCaching, test_index_descriptor_getters)
{
    const read_id_t first_read      = 15;
    const read_id_t number_of_reads = 156;
    const IndexDescriptor index_descriptor(first_read, number_of_reads);

    ASSERT_EQ(index_descriptor.first_read(), first_read);
    ASSERT_EQ(index_descriptor.number_of_reads(), number_of_reads);
}

TEST(TestCudamapperIndexCaching, test_index_descriptor_equality_operators)
{
    const IndexDescriptor index_descriptor_15_156_1(15, 156);
    const IndexDescriptor index_descriptor_15_156_2(15, 156);
    const IndexDescriptor index_descriptor_16_156(16, 156);
    const IndexDescriptor index_descriptor_15_157(15, 157);
    const IndexDescriptor index_descriptor_16_157(16, 157);

    ASSERT_EQ(index_descriptor_15_156_1, index_descriptor_15_156_2);
    ASSERT_NE(index_descriptor_15_156_1, index_descriptor_16_156);
    ASSERT_NE(index_descriptor_15_156_1, index_descriptor_15_157);
    ASSERT_NE(index_descriptor_15_156_1, index_descriptor_16_157);
}

TEST(TestCudamapperIndexCaching, test_index_descriptor_hash)
{
    const read_id_t first_read      = 0x24;
    const read_id_t number_of_reads = 0xCF;
    const IndexDescriptor index_descriptor(first_read, number_of_reads);
    std::size_t hash = 0;
    if (sizeof(std::size_t) == 4)
    {
        hash |= 0xCF'00'24;
    }
    else if (sizeof(std::size_t) == 8)
    {
        hash |= 0xCF'00'00'00'24;
    }
    else
    {
        ASSERT_TRUE(false); // implement test for system where std::size_t is not 32 or 64 bits
    }
    const IndexDescriptorHash index_descriptor_hash;
    ASSERT_EQ(index_descriptor_hash(index_descriptor), hash);
}

} // namespace cudamapper
} // namespace claragenomics