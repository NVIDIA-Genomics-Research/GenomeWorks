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

#include "../src/index_descriptor.hpp"

#include "cudamapper_file_location.hpp"

namespace claragenomics
{
namespace cudamapper
{

// *** test IndexDescriptor and IndexDescriptorHash ***

TEST(TestCudamapperIndexDescriptor, test_index_descriptor_getters)
{
    const read_id_t first_read              = 15;
    const number_of_reads_t number_of_reads = 156;
    const IndexDescriptor index_descriptor(first_read, number_of_reads);

    ASSERT_EQ(index_descriptor.first_read(), first_read);
    ASSERT_EQ(index_descriptor.number_of_reads(), number_of_reads);
}

TEST(TestCudamapperIndexDescriptor, test_index_descriptor_equality_operators)
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

TEST(TestCudamapperIndexDescriptor, test_index_descriptor_hash)
{
    static_assert(sizeof(size_t) == 8, "only 64-bit values supported, adjust element_mask and shift_bits");
    const read_id_t first_read              = 0x24;
    const number_of_reads_t number_of_reads = 0xCF;
    const IndexDescriptor index_descriptor(first_read, number_of_reads);
    std::size_t hash = 0xCF'00'00'00'24;
    const IndexDescriptorHash index_descriptor_hash;
    ASSERT_EQ(index_descriptor_hash(index_descriptor), hash);
}

/// *** test group_reads_into_indices ***

void test_group_reads_into_indices(const io::FastaParser& parser,
                                   const number_of_basepairs_t max_basepairs_per_index,
                                   const std::vector<IndexDescriptor>& expected_index_descriptors)
{
    const std::vector<IndexDescriptor>& generated_index_descriptors = group_reads_into_indices(parser,
                                                                                               max_basepairs_per_index);

    ASSERT_EQ(expected_index_descriptors.size(), generated_index_descriptors.size());

    for (std::size_t i = 0; i < expected_index_descriptors.size(); ++i)
    {
        const IndexDescriptor& expected_index_descriptor  = expected_index_descriptors[i];
        const IndexDescriptor& generated_index_descriptor = generated_index_descriptors[i];
        ASSERT_EQ(expected_index_descriptor.first_read(), generated_index_descriptor.first_read()) << "i: " << i;
        ASSERT_EQ(expected_index_descriptor.number_of_reads(), generated_index_descriptor.number_of_reads()) << "i: " << i;
    }
}

TEST(TestCudamapperIndexDescriptor, test_group_reads_into_indices_all_reads_fit_in_indices)
{
    // target FASTA: 20_reads.fasta
    // read_0:  4 basepairs
    // read_1:  6 basepairs
    // read_2:  7 basepairs
    // read_3:  4 basepairs
    // read_4:  3 basepairs
    // read_5:  8 basepairs
    // read_6:  6 basepairs
    // read_7:  3 basepairs
    // read_8:  3 basepairs
    // read_9:  5 basepairs
    // read_10: 7 basepairs
    // read_11: 3 basepairs
    // read_12: 2 basepairs
    // read_13: 4 basepairs
    // read_14: 4 basepairs
    // read_15: 2 basepairs
    // read_16: 5 basepairs
    // read_17: 6 basepairs
    // read_18: 2 basepairs
    // read_19: 4 basepairs
    // max basepairs in index: 10
    // indices: {0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}, {8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}, {19, 1}

    const std::shared_ptr<const io::FastaParser> parser = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta", 1, false);
    const number_of_basepairs_t max_basepairs_per_index = 10;

    const std::vector<IndexDescriptor> expected_index_descriptors = {{0, 2}, {2, 1}, {3, 2}, {5, 1}, {6, 2}, {8, 2}, {10, 2}, {12, 3}, {15, 2}, {17, 2}, {19, 1}};

    test_group_reads_into_indices(*parser,
                                  max_basepairs_per_index,
                                  expected_index_descriptors);
}

TEST(TestCudamapperIndexDescriptor, test_group_reads_into_indices_some_reads_larger_than_index)
{
    // Some reads have more basepairs than max_basepairs_per_index
    // In that case those reads should be place in an index alone

    // target FASTA: 20_reads.fasta
    // read_0:  4 basepairs
    // read_1:  6 basepairs
    // read_2:  7 basepairs
    // read_3:  4 basepairs
    // read_4:  3 basepairs
    // read_5:  8 basepairs
    // read_6:  6 basepairs
    // read_7:  3 basepairs
    // read_8:  3 basepairs
    // read_9:  5 basepairs
    // read_10: 7 basepairs
    // read_11: 3 basepairs
    // read_12: 2 basepairs
    // read_13: 4 basepairs
    // read_14: 4 basepairs
    // read_15: 2 basepairs
    // read_16: 5 basepairs
    // read_17: 6 basepairs
    // read_18: 2 basepairs
    // read_19: 4 basepairs
    // max basepairs in index: 7
    // indices: {0, 1}, {1, 1}, {2, 1}, {3, 2}, {5, 1}, {6, 1}, {7, 2}, {9, 1}, {10, 1}, {11, 2}, {13, 1}, {14, 2}, {16, 1}, {17, 1}, {18, 2}

    const std::shared_ptr<const io::FastaParser> parser = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/20_reads.fasta", 1, false);
    const number_of_basepairs_t max_basepairs_per_index = 7;

    const std::vector<IndexDescriptor> expected_index_descriptors = {{0, 1}, {1, 1}, {2, 1}, {3, 2}, {5, 1}, {6, 1}, {7, 2}, {9, 1}, {10, 1}, {11, 2}, {13, 1}, {14, 2}, {16, 1}, {17, 1}, {18, 2}};

    test_group_reads_into_indices(*parser,
                                  max_basepairs_per_index,
                                  expected_index_descriptors);
}

} // namespace cudamapper
} // namespace claragenomics
