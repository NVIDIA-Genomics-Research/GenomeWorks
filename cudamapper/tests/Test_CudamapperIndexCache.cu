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

#include <string>
#include <vector>

#include "cudamapper_file_location.hpp"

#include <claragenomics/utils/signed_integer_utils.hpp>
#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/io/fasta_parser.hpp>

namespace claragenomics
{
namespace cudamapper
{

// *** Test IndexCacheHost ***

void check_if_index_is_correct(const std::shared_ptr<Index>& index,
                               const std::vector<representation_t>& expected_representations,
                               const std::vector<read_id_t>& expected_read_ids,
                               const std::vector<position_in_read_t>& expected_positions_in_reads,
                               const std::vector<SketchElement::DirectionOfRepresentation>& expected_directions_of_reads,
                               const std::vector<representation_t>& expected_unique_representations,
                               const std::vector<std::uint32_t>& expected_first_occurrence_of_representations,
                               const read_id_t expected_number_of_reads,
                               const read_id_t expected_smallest_read_id,
                               const read_id_t expected_largest_read_id,
                               const position_in_read_t expected_number_of_basepairs_in_longest_read,
                               const uint64_t expected_maximum_kmer_size,
                               const cudaStream_t cuda_stream,
                               const std::string& test_uid)
{
    ASSERT_EQ(get_size(expected_representations), get_size(expected_read_ids)) << " test_uid: " << test_uid;
    ASSERT_EQ(get_size(expected_representations), get_size(expected_positions_in_reads)) << " test_uid: " << test_uid;
    ASSERT_EQ(get_size(expected_representations), get_size(expected_directions_of_reads)) << " test_uid: " << test_uid;
    ASSERT_EQ(get_size(expected_unique_representations), get_size(expected_first_occurrence_of_representations) - 1) << " test_uid: " << test_uid;
    ASSERT_LE(expected_number_of_reads, expected_largest_read_id - expected_smallest_read_id + 1) << " test_uid: " << test_uid;

    ASSERT_EQ(get_size(index->representations()), get_size(expected_representations)) << " test_uid: " << test_uid;
    std::vector<representation_t> index_representations(index->representations().size());
    cudautils::device_copy_n(index->representations().data(), index->representations().size(), index_representations.data(), cuda_stream); // D2H

    ASSERT_EQ(get_size(index->read_ids()), get_size(expected_read_ids)) << " test_uid: " << test_uid;
    std::vector<read_id_t> index_read_ids(index->read_ids().size());
    cudautils::device_copy_n(index->read_ids().data(), index->read_ids().size(), index_read_ids.data(), cuda_stream); // D2H

    ASSERT_EQ(get_size(index->positions_in_reads()), get_size(expected_positions_in_reads)) << " test_uid: " << test_uid;
    std::vector<position_in_read_t> index_positions_in_reads(index->positions_in_reads().size());
    cudautils::device_copy_n(index->positions_in_reads().data(), index->positions_in_reads().size(), index_positions_in_reads.data(), cuda_stream); // D2H

    ASSERT_EQ(get_size(index->directions_of_reads()), get_size(expected_directions_of_reads)) << " test_uid: " << test_uid;
    std::vector<SketchElement::DirectionOfRepresentation> index_directions_of_reads(index->directions_of_reads().size());
    cudautils::device_copy_n(index->directions_of_reads().data(), index->directions_of_reads().size(), index_directions_of_reads.data(), cuda_stream); // D2H

    ASSERT_EQ(get_size(index->unique_representations()), get_size(expected_unique_representations)) << " test_uid: " << test_uid;
    std::vector<representation_t> index_unique_representations(index->unique_representations().size());
    cudautils::device_copy_n(index->unique_representations().data(), index->unique_representations().size(), index_unique_representations.data(), cuda_stream); // D2H

    ASSERT_EQ(get_size(index->first_occurrence_of_representations()), get_size(expected_first_occurrence_of_representations)) << " test_uid: " << test_uid;
    std::vector<std::uint32_t> index_first_occurrence_of_representations(index->first_occurrence_of_representations().size());
    cudautils::device_copy_n(index->first_occurrence_of_representations().data(), index->first_occurrence_of_representations().size(), index_first_occurrence_of_representations.data(), cuda_stream); // D2H

    read_id_t index_number_of_reads = index->number_of_reads();

    read_id_t index_smallest_read_id = index->smallest_read_id();

    read_id_t index_largest_read_id = index->largest_read_id();

    position_in_read_t index_number_of_basepairs_in_longest_read = index->number_of_basepairs_in_longest_read();

    uint64_t index_maximum_kmer_size = index->maximum_kmer_size();

    CGA_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));

    ASSERT_EQ(index_representations, expected_representations) << " test_uid: " << test_uid;
    ASSERT_EQ(index_read_ids, expected_read_ids) << " test_uid: " << test_uid;
    ASSERT_EQ(index_positions_in_reads, expected_positions_in_reads) << " test_uid: " << test_uid;
    ASSERT_EQ(index_directions_of_reads, expected_directions_of_reads) << " test_uid: " << test_uid;
    ASSERT_EQ(index_unique_representations, expected_unique_representations) << " test_uid: " << test_uid;
    ASSERT_EQ(index_first_occurrence_of_representations, expected_first_occurrence_of_representations) << " test_uid: " << test_uid;
    ASSERT_EQ(index_number_of_reads, expected_number_of_reads) << " test_uid: " << test_uid;
    ASSERT_EQ(index_smallest_read_id, expected_smallest_read_id) << " test_uid: " << test_uid;
    ASSERT_EQ(index_largest_read_id, expected_largest_read_id) << " test_uid: " << test_uid;
    ASSERT_EQ(index_number_of_basepairs_in_longest_read, expected_number_of_basepairs_in_longest_read) << " test_uid: " << test_uid;
    ASSERT_EQ(index_maximum_kmer_size, expected_maximum_kmer_size) << " test_uid: " << test_uid;
}

TEST(TestCudamapperIndexCaching, test_index_cache_host_same_query_and_target)
{
    // catcaag_aagcta.fasta k = 3 w = 2

    // >read_0
    // CATCAAG
    // >read_1
    // AAGCTA

    // ** CATCAAG **

    // kmer representation: forward, reverse
    // CAT:  103 <032>
    // ATC: <031> 203
    // TCA: <310> 320
    // CAA: <100> 332
    // AAG: <002> 133

    // front end minimizers: representation, position_in_read, direction, read_id
    // CAT: 032 0 R 0

    // central minimizers
    // CATC: 031 1 F 0
    // ATCA: 031 1 F 0
    // TCAA: 100 3 F 0
    // CAAG: 002 4 F 0

    // back end minimizers
    // AAG: 002 4 F 0

    // All minimizers: AAG(4f), ATC(1f), ATG(0r), CAA(3f)

    // ** AAGCTA **

    // kmer representation: forward, reverse
    // AAG: <002> 133
    // AGC: <021> 213
    // GCT:  213 <021>
    // CTA: <130> 302

    // front end minimizers: representation, position_in_read, direction, read_id
    // AAG: 002 0 F 1

    // central minimizers
    // AAGC: 002 0 F 1
    // AGCT: 021 2 R 1 // only the last minimizer is saved
    // GCTA: 021 2 R 1

    // back end minimizers
    // CTA: 130 3 F 1

    // All minimizers: AAG(0f), AGC(2r), CTA(3f)

    // CATCAAG_AAGCTA
    // All minimizers: AAG(4f0), AAG(0f1), AGC(2r1), ATC(1f0), ATG(0r0), CAA(3f0), CTA(3f1)

    cudaStream_t cuda_stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));

    const bool same_query_and_target               = true;
    std::shared_ptr<io::FastaParser> query_parser  = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta");
    std::shared_ptr<io::FastaParser> target_parser = query_parser;
    DefaultDeviceAllocator allocator               = create_default_device_allocator();
    const std::uint64_t k                          = 3;
    const std::uint64_t w                          = 2;
    const bool hash_representations                = false;
    const double filtering_parameter               = 1.0;

    // ************* expected indices *************

    // ** CATCAAG: AAG(4f), ATC(1f), CAA(3f), ATG(0r)
    std::vector<representation_t> catcaag_representations;
    std::vector<read_id_t> catcaag_read_ids;
    std::vector<position_in_read_t> catcaag_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> catcaag_directions_of_reads;
    std::vector<representation_t> catcaag_unique_representations;
    std::vector<std::uint32_t> catcaag_first_occurrence_of_representations;

    // AAG(4f)
    catcaag_representations.push_back(0b000010);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(4);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b000010);
    catcaag_first_occurrence_of_representations.push_back(0);
    // ATC(1f)
    catcaag_representations.push_back(0b001101);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(1);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b001101);
    catcaag_first_occurrence_of_representations.push_back(1);
    // ATG(0r)
    catcaag_representations.push_back(0b001110);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(0);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    catcaag_unique_representations.push_back(0b001110);
    catcaag_first_occurrence_of_representations.push_back(2);
    // CAA(3f)
    catcaag_representations.push_back(0b010000);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(3);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b010000);
    catcaag_first_occurrence_of_representations.push_back(3);
    // trailing elements
    catcaag_first_occurrence_of_representations.push_back(4);

    const read_id_t catcaag_number_of_reads = 1;
    const std::vector<std::string> catcaag_read_ids_to_read_names({"read_0"});
    const std::vector<std::uint32_t> catcaag_read_ids_to_read_lengths({7});
    const read_id_t catcaag_smallest_read_id                             = 0;
    const read_id_t catcaag_largest_read_id                              = 0;
    const position_in_read_t catcaag_number_of_basepairs_in_longest_read = 7;
    const uint64_t catcaag_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ** AAGCTA: AAG(0f), AGC(2f), CTA(3f)
    std::vector<representation_t> aagcta_representations;
    std::vector<read_id_t> aagcta_read_ids;
    std::vector<position_in_read_t> aagcta_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> aagcta_directions_of_reads;
    std::vector<representation_t> aagcta_unique_representations;
    std::vector<std::uint32_t> aagcta_first_occurrence_of_representations;

    // AAG(0f)
    aagcta_representations.push_back(0b000010);
    aagcta_read_ids.push_back(1);
    aagcta_positions_in_reads.push_back(0);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b000010);
    aagcta_first_occurrence_of_representations.push_back(0);
    // AGC(2r)
    aagcta_representations.push_back(0b001001);
    aagcta_read_ids.push_back(1);
    aagcta_positions_in_reads.push_back(2);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    aagcta_unique_representations.push_back(0b001001);
    aagcta_first_occurrence_of_representations.push_back(1);
    // CTA(3f)
    aagcta_representations.push_back(0b011100);
    aagcta_read_ids.push_back(1);
    aagcta_positions_in_reads.push_back(3);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b011100);
    aagcta_first_occurrence_of_representations.push_back(2);
    // trailing elements
    aagcta_first_occurrence_of_representations.push_back(3);

    const read_id_t aagcta_number_of_reads                              = 1;
    const read_id_t aagcta_smallest_read_id                             = 1;
    const read_id_t aagcta_largest_read_id                              = 1;
    const position_in_read_t aagcta_number_of_basepairs_in_longest_read = 6;
    const uint64_t aagcta_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ** CATCAAG_AAGCTA: AAG(4f0), AAG(0f1), AGC(2r1), ATC(1f0), ATG(0r0), CAA(3f0), CTA(3f1)
    std::vector<representation_t> catcaag_aagcta_representations;
    std::vector<read_id_t> catcaag_aagcta_read_ids;
    std::vector<position_in_read_t> catcaag_aagcta_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> catcaag_aagcta_directions_of_reads;
    std::vector<representation_t> catcaag_aagcta_unique_representations;
    std::vector<std::uint32_t> catcaag_aagcta_first_occurrence_of_representations;

    // AAG(4f0)
    catcaag_aagcta_representations.push_back(0b000010);
    catcaag_aagcta_read_ids.push_back(0);
    catcaag_aagcta_positions_in_reads.push_back(4);
    catcaag_aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_aagcta_unique_representations.push_back(0b000010);
    catcaag_aagcta_first_occurrence_of_representations.push_back(0);
    // AAG(0f1)
    catcaag_aagcta_representations.push_back(0b000010);
    catcaag_aagcta_read_ids.push_back(1);
    catcaag_aagcta_positions_in_reads.push_back(0);
    catcaag_aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    // AGC(2r1)
    catcaag_aagcta_representations.push_back(0b001001);
    catcaag_aagcta_read_ids.push_back(1);
    catcaag_aagcta_positions_in_reads.push_back(2);
    catcaag_aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    catcaag_aagcta_unique_representations.push_back(0b001001);
    catcaag_aagcta_first_occurrence_of_representations.push_back(2);
    // ATC(1f0)
    catcaag_aagcta_representations.push_back(0b001101);
    catcaag_aagcta_read_ids.push_back(0);
    catcaag_aagcta_positions_in_reads.push_back(1);
    catcaag_aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_aagcta_unique_representations.push_back(0b001101);
    catcaag_aagcta_first_occurrence_of_representations.push_back(3);
    // ATG(0r0)
    catcaag_aagcta_representations.push_back(0b001110);
    catcaag_aagcta_read_ids.push_back(0);
    catcaag_aagcta_positions_in_reads.push_back(0);
    catcaag_aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    catcaag_aagcta_unique_representations.push_back(0b001110);
    catcaag_aagcta_first_occurrence_of_representations.push_back(4);
    // CAA(3f0)
    catcaag_aagcta_representations.push_back(0b010000);
    catcaag_aagcta_read_ids.push_back(0);
    catcaag_aagcta_positions_in_reads.push_back(3);
    catcaag_aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_aagcta_unique_representations.push_back(0b010000);
    catcaag_aagcta_first_occurrence_of_representations.push_back(5);
    // CTA(3f1)
    catcaag_aagcta_representations.push_back(0b011100);
    catcaag_aagcta_read_ids.push_back(1);
    catcaag_aagcta_positions_in_reads.push_back(3);
    catcaag_aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_aagcta_unique_representations.push_back(0b011100);
    catcaag_aagcta_first_occurrence_of_representations.push_back(6);
    // trailing elements
    catcaag_aagcta_first_occurrence_of_representations.push_back(7);

    const read_id_t catcaag_aagcta_number_of_reads                              = 2;
    const read_id_t catcaag_aagcta_smallest_read_id                             = 0;
    const read_id_t catcaag_aagcta_largest_read_id                              = 1;
    const position_in_read_t catcaag_aagcta_number_of_basepairs_in_longest_read = 7;
    const uint64_t catcaag_aagcta_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ************* IndexCacheHost tests *************

    IndexDescriptor catcaag_index_descriptor(0, 1);
    IndexDescriptor aagcta_index_descriptor(1, 1);
    IndexDescriptor catcaag_aagcta_index_descriptor(0, 2);
    std::vector<IndexDescriptor> catcaag_index_descriptors({catcaag_index_descriptor});
    std::vector<IndexDescriptor> aagcta_index_descriptors({aagcta_index_descriptor});
    std::vector<IndexDescriptor> catcaag_aagcta_separate_index_descriptors({catcaag_index_descriptor, aagcta_index_descriptor});
    std::vector<IndexDescriptor> catcaag_aagcta_one_index_descriptors({catcaag_aagcta_index_descriptor});

    IndexCacheHost index_host_cache(same_query_and_target,
                                    allocator,
                                    query_parser,
                                    target_parser,
                                    k,
                                    w,
                                    hash_representations,
                                    filtering_parameter,
                                    cuda_stream);

    index_host_cache.generate_query_cache_content(catcaag_index_descriptors);

    auto index_query_catcaag = index_host_cache.get_index_from_query_cache(catcaag_index_descriptor);
    check_if_index_is_correct(index_query_catcaag,
                              catcaag_representations,
                              catcaag_read_ids,
                              catcaag_positions_in_reads,
                              catcaag_directions_of_reads,
                              catcaag_unique_representations,
                              catcaag_first_occurrence_of_representations,
                              catcaag_number_of_reads,
                              catcaag_smallest_read_id,
                              catcaag_largest_read_id,
                              catcaag_number_of_basepairs_in_longest_read,
                              catcaag_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_1");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_target_cache(catcaag_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_target_cache(aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));

    index_host_cache.generate_target_cache_content(aagcta_index_descriptors);

    index_query_catcaag = index_host_cache.get_index_from_query_cache(catcaag_index_descriptor);
    check_if_index_is_correct(index_query_catcaag,
                              catcaag_representations,
                              catcaag_read_ids,
                              catcaag_positions_in_reads,
                              catcaag_directions_of_reads,
                              catcaag_unique_representations,
                              catcaag_first_occurrence_of_representations,
                              catcaag_number_of_reads,
                              catcaag_smallest_read_id,
                              catcaag_largest_read_id,
                              catcaag_number_of_basepairs_in_longest_read,
                              catcaag_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_2");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_target_cache(catcaag_index_descriptor));
    auto index_target_aagcta = index_host_cache.get_index_from_target_cache(aagcta_index_descriptor);
    check_if_index_is_correct(index_target_aagcta,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_3");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));

    index_host_cache.generate_query_cache_content(aagcta_index_descriptors);

    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_index_descriptor));
    auto index_query_aagcta = index_host_cache.get_index_from_query_cache(aagcta_index_descriptor);
    check_if_index_is_correct(index_query_aagcta,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_4");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_target_cache(catcaag_index_descriptor));
    index_target_aagcta = index_host_cache.get_index_from_target_cache(aagcta_index_descriptor);
    check_if_index_is_correct(index_target_aagcta,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_5");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));

    index_host_cache.generate_query_cache_content(catcaag_aagcta_separate_index_descriptors);

    auto index_query_catcaag_separate = index_host_cache.get_index_from_query_cache(catcaag_index_descriptor);
    check_if_index_is_correct(index_query_catcaag_separate,
                              catcaag_representations,
                              catcaag_read_ids,
                              catcaag_positions_in_reads,
                              catcaag_directions_of_reads,
                              catcaag_unique_representations,
                              catcaag_first_occurrence_of_representations,
                              catcaag_number_of_reads,
                              catcaag_smallest_read_id,
                              catcaag_largest_read_id,
                              catcaag_number_of_basepairs_in_longest_read,
                              catcaag_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_6");
    auto index_query_aagcta_separate = index_host_cache.get_index_from_query_cache(aagcta_index_descriptor);
    check_if_index_is_correct(index_query_aagcta_separate,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_7");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_target_cache(catcaag_index_descriptor));
    index_target_aagcta = index_host_cache.get_index_from_target_cache(aagcta_index_descriptor);
    check_if_index_is_correct(index_target_aagcta,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_8");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));

    index_host_cache.generate_target_cache_content(catcaag_aagcta_one_index_descriptors);

    index_query_catcaag_separate = index_host_cache.get_index_from_query_cache(catcaag_index_descriptor);
    check_if_index_is_correct(index_query_catcaag_separate,
                              catcaag_representations,
                              catcaag_read_ids,
                              catcaag_positions_in_reads,
                              catcaag_directions_of_reads,
                              catcaag_unique_representations,
                              catcaag_first_occurrence_of_representations,
                              catcaag_number_of_reads,
                              catcaag_smallest_read_id,
                              catcaag_largest_read_id,
                              catcaag_number_of_basepairs_in_longest_read,
                              catcaag_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_9");
    index_query_aagcta_separate = index_host_cache.get_index_from_query_cache(aagcta_index_descriptor);
    check_if_index_is_correct(index_query_aagcta_separate,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_10");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_query_cache(catcaag_aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_target_cache(catcaag_index_descriptor));
    ASSERT_ANY_THROW(index_host_cache.get_index_from_target_cache(aagcta_index_descriptor));
    auto catcaag_aagcta_target_aagcta = index_host_cache.get_index_from_target_cache(catcaag_aagcta_index_descriptor);
    check_if_index_is_correct(catcaag_aagcta_target_aagcta,
                              catcaag_aagcta_representations,
                              catcaag_aagcta_read_ids,
                              catcaag_aagcta_positions_in_reads,
                              catcaag_aagcta_directions_of_reads,
                              catcaag_aagcta_unique_representations,
                              catcaag_aagcta_first_occurrence_of_representations,
                              catcaag_aagcta_number_of_reads,
                              catcaag_aagcta_smallest_read_id,
                              catcaag_aagcta_largest_read_id,
                              catcaag_aagcta_number_of_basepairs_in_longest_read,
                              catcaag_aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_same_query_and_target_11");

    CGA_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    CGA_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperIndexCaching, test_index_cache_host_not_the_same_query_and_target)
{
    // aagcta.fasta ctacaag.fasta k = 3 w = 2

    // ** AAGCTA **

    // kmer representation: forward, reverse
    // AAG: <002> 133
    // AGC: <021> 213
    // GCT:  213 <021>
    // CTA: <130> 302

    // front end minimizers: representation, position_in_read, direction, read_id
    // AAG: 002 0 F 1

    // central minimizers
    // AAGC: 002 0 F 1
    // AGCT: 021 2 R 1 // only the last minimizer is saved
    // GCTA: 021 2 R 1

    // back end minimizers
    // CTA: 130 3 F 1

    // All minimizers: AAG(0f), AGC(2r), CTA(3f)

    // ** CATCAAG **

    // kmer representation: forward, reverse
    // CAT:  103 <032>
    // ATC: <031> 203
    // TCA: <310> 320
    // CAA: <100> 332
    // AAG: <002> 133

    // front end minimizers: representation, position_in_read, direction, read_id
    // CAT: 032 0 R 0

    // central minimizers
    // CATC: 031 1 F 0
    // ATCA: 031 1 F 0
    // TCAA: 100 3 F 0
    // CAAG: 002 4 F 0

    // back end minimizers
    // AAG: 002 4 F 0

    // All minimizers: AAG(4f), ATC(1f), ATG(0r), CAA(3f)

    cudaStream_t cuda_stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));

    const bool same_query_and_target               = false;
    std::shared_ptr<io::FastaParser> query_parser  = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/aagcta.fasta");
    std::shared_ptr<io::FastaParser> target_parser = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag.fasta");
    DefaultDeviceAllocator allocator               = create_default_device_allocator();
    const std::uint64_t k                          = 3;
    const std::uint64_t w                          = 2;
    const bool hash_representations                = false;
    const double filtering_parameter               = 1.0;

    // ************* expected indices *************

    // ** AAGCTA: AAG(0f), AGC(2f), CTA(3f)
    std::vector<representation_t> aagcta_representations;
    std::vector<read_id_t> aagcta_read_ids;
    std::vector<position_in_read_t> aagcta_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> aagcta_directions_of_reads;
    std::vector<representation_t> aagcta_unique_representations;
    std::vector<std::uint32_t> aagcta_first_occurrence_of_representations;

    // AAG(0f)
    aagcta_representations.push_back(0b000010);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(0);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b000010);
    aagcta_first_occurrence_of_representations.push_back(0);
    // AGC(2r)
    aagcta_representations.push_back(0b001001);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(2);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    aagcta_unique_representations.push_back(0b001001);
    aagcta_first_occurrence_of_representations.push_back(1);
    // CTA(3f)
    aagcta_representations.push_back(0b011100);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(3);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b011100);
    aagcta_first_occurrence_of_representations.push_back(2);
    // trailing elements
    aagcta_first_occurrence_of_representations.push_back(3);

    const read_id_t aagcta_number_of_reads                              = 1;
    const read_id_t aagcta_smallest_read_id                             = 0;
    const read_id_t aagcta_largest_read_id                              = 0;
    const position_in_read_t aagcta_number_of_basepairs_in_longest_read = 6;
    const uint64_t aagcta_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ** CATCAAG: AAG(4f), ATC(1f), CAA(3f), ATG(0r)
    std::vector<representation_t> catcaag_representations;
    std::vector<read_id_t> catcaag_read_ids;
    std::vector<position_in_read_t> catcaag_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> catcaag_directions_of_reads;
    std::vector<representation_t> catcaag_unique_representations;
    std::vector<std::uint32_t> catcaag_first_occurrence_of_representations;

    // AAG(4f)
    catcaag_representations.push_back(0b000010);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(4);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b000010);
    catcaag_first_occurrence_of_representations.push_back(0);
    // ATC(1f)
    catcaag_representations.push_back(0b001101);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(1);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b001101);
    catcaag_first_occurrence_of_representations.push_back(1);
    // ATG(0r)
    catcaag_representations.push_back(0b001110);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(0);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    catcaag_unique_representations.push_back(0b001110);
    catcaag_first_occurrence_of_representations.push_back(2);
    // CAA(3f)
    catcaag_representations.push_back(0b010000);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(3);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b010000);
    catcaag_first_occurrence_of_representations.push_back(3);
    // trailing elements
    catcaag_first_occurrence_of_representations.push_back(4);

    const read_id_t catcaag_number_of_reads                              = 1;
    const read_id_t catcaag_smallest_read_id                             = 0;
    const read_id_t catcaag_largest_read_id                              = 0;
    const position_in_read_t catcaag_number_of_basepairs_in_longest_read = 7;
    const uint64_t catcaag_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ************* IndexCacheHost tests *************

    // both descriptors are the same, but they are going to be used with different parsers
    IndexDescriptor index_descriptor(0, 1);
    std::vector<IndexDescriptor> index_descriptors({index_descriptor});

    IndexCacheHost index_host_cache(same_query_and_target,
                                    allocator,
                                    query_parser,
                                    target_parser,
                                    k,
                                    w,
                                    hash_representations,
                                    filtering_parameter,
                                    cuda_stream);

    index_host_cache.generate_query_cache_content(index_descriptors);

    auto index_query_aagcta = index_host_cache.get_index_from_query_cache(index_descriptor);
    check_if_index_is_correct(index_query_aagcta,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_not_the_same_query_and_target_1");
    ASSERT_ANY_THROW(index_host_cache.get_index_from_target_cache(index_descriptor));

    index_host_cache.generate_target_cache_content(index_descriptors);

    index_query_aagcta = index_host_cache.get_index_from_query_cache(index_descriptor);
    check_if_index_is_correct(index_query_aagcta,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_not_the_same_query_and_target_2");
    auto index_target_catcaag = index_host_cache.get_index_from_target_cache(index_descriptor);
    check_if_index_is_correct(index_target_catcaag,
                              catcaag_representations,
                              catcaag_read_ids,
                              catcaag_positions_in_reads,
                              catcaag_directions_of_reads,
                              catcaag_unique_representations,
                              catcaag_first_occurrence_of_representations,
                              catcaag_number_of_reads,
                              catcaag_smallest_read_id,
                              catcaag_largest_read_id,
                              catcaag_number_of_basepairs_in_longest_read,
                              catcaag_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_not_the_same_query_and_target_3");

    CGA_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    CGA_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperIndexCaching, test_index_cache_host_keep_on_device)
{
    // AAGCTA: AAG(0f), AGC(2r), CTA(3f)

    cudaStream_t cuda_stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));

    const bool same_query_and_target               = true;
    std::shared_ptr<io::FastaParser> query_parser  = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/aagcta.fasta");
    std::shared_ptr<io::FastaParser> target_parser = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/aagcta.fasta");
    DefaultDeviceAllocator allocator               = create_default_device_allocator();
    const std::uint64_t k                          = 3;
    const std::uint64_t w                          = 2;
    const bool hash_representations                = false;
    const double filtering_parameter               = 1.0;

    // ************* expected indices *************

    // ** AAGCTA: AAG(0f), AGC(2f), CTA(3f)
    std::vector<representation_t> aagcta_representations;
    std::vector<read_id_t> aagcta_read_ids;
    std::vector<position_in_read_t> aagcta_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> aagcta_directions_of_reads;
    std::vector<representation_t> aagcta_unique_representations;
    std::vector<std::uint32_t> aagcta_first_occurrence_of_representations;

    // AAG(0f)
    aagcta_representations.push_back(0b000010);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(0);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b000010);
    aagcta_first_occurrence_of_representations.push_back(0);
    // AGC(2r)
    aagcta_representations.push_back(0b001001);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(2);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    aagcta_unique_representations.push_back(0b001001);
    aagcta_first_occurrence_of_representations.push_back(1);
    // CTA(3f)
    aagcta_representations.push_back(0b011100);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(3);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b011100);
    aagcta_first_occurrence_of_representations.push_back(2);
    // trailing elements
    aagcta_first_occurrence_of_representations.push_back(3);

    const read_id_t aagcta_number_of_reads                              = 1;
    const read_id_t aagcta_smallest_read_id                             = 0;
    const read_id_t aagcta_largest_read_id                              = 0;
    const position_in_read_t aagcta_number_of_basepairs_in_longest_read = 6;
    const uint64_t aagcta_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ************* IndexCacheDevice tests *************

    IndexDescriptor index_descriptor(0, 1);
    std::vector<IndexDescriptor> index_descriptors({index_descriptor});

    auto index_cache_host = std::make_shared<IndexCacheHost>(same_query_and_target,
                                                             allocator,
                                                             query_parser,
                                                             target_parser,
                                                             k,
                                                             w,
                                                             hash_representations,
                                                             filtering_parameter,
                                                             cuda_stream);

    index_cache_host->generate_query_cache_content(index_descriptors,
                                                   index_descriptors);
    index_cache_host->generate_target_cache_content(index_descriptors,
                                                    index_descriptors);

    auto index_query_temp_device_cache  = index_cache_host->get_index_from_query_cache(index_descriptor);
    auto index_query_copy_from_host     = index_cache_host->get_index_from_query_cache(index_descriptor);
    auto index_target_temp_device_cache = index_cache_host->get_index_from_target_cache(index_descriptor);
    auto index_target_copy_from_host    = index_cache_host->get_index_from_target_cache(index_descriptor);

    ASSERT_EQ(index_query_temp_device_cache, index_target_temp_device_cache);
    ASSERT_NE(index_query_temp_device_cache, index_query_copy_from_host);
    ASSERT_NE(index_target_temp_device_cache, index_target_copy_from_host);

    check_if_index_is_correct(index_query_temp_device_cache,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_keep_on_device_1");
    check_if_index_is_correct(index_query_copy_from_host,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_keep_on_device_2");
    check_if_index_is_correct(index_target_temp_device_cache,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_keep_on_device_3");
    check_if_index_is_correct(index_target_copy_from_host,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_host_keep_on_device_4");
}

// *** Test IndexCacheDevice ***

TEST(TestCudamapperIndexCaching, test_index_cache_device_same_query_and_target)
{
    // >read_0
    // CATCAAG
    // >read_1
    // AAGCTA

    // CATCAAG minimizers: AAG(4f), ATC(1f), ATG(0r), CAA(3f)
    // AAGCTA minimizers: AAG(0f), AGC(2r), CTA(3f)

    cudaStream_t cuda_stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));

    const bool same_query_and_target               = true;
    std::shared_ptr<io::FastaParser> query_parser  = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta");
    std::shared_ptr<io::FastaParser> target_parser = query_parser;
    DefaultDeviceAllocator allocator               = create_default_device_allocator();
    const std::uint64_t k                          = 3;
    const std::uint64_t w                          = 2;
    const bool hash_representations                = false;
    const double filtering_parameter               = 1.0;

    // ************* expected indices *************

    // ** CATCAAG: AAG(4f), ATC(1f), CAA(3f), ATG(0r)
    std::vector<representation_t> catcaag_representations;
    std::vector<read_id_t> catcaag_read_ids;
    std::vector<position_in_read_t> catcaag_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> catcaag_directions_of_reads;
    std::vector<representation_t> catcaag_unique_representations;
    std::vector<std::uint32_t> catcaag_first_occurrence_of_representations;

    // AAG(4f)
    catcaag_representations.push_back(0b000010);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(4);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b000010);
    catcaag_first_occurrence_of_representations.push_back(0);
    // ATC(1f)
    catcaag_representations.push_back(0b001101);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(1);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b001101);
    catcaag_first_occurrence_of_representations.push_back(1);
    // ATG(0r)
    catcaag_representations.push_back(0b001110);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(0);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    catcaag_unique_representations.push_back(0b001110);
    catcaag_first_occurrence_of_representations.push_back(2);
    // CAA(3f)
    catcaag_representations.push_back(0b010000);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(3);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b010000);
    catcaag_first_occurrence_of_representations.push_back(3);
    // trailing elements
    catcaag_first_occurrence_of_representations.push_back(4);

    const read_id_t catcaag_number_of_reads                              = 1;
    const read_id_t catcaag_smallest_read_id                             = 0;
    const read_id_t catcaag_largest_read_id                              = 0;
    const position_in_read_t catcaag_number_of_basepairs_in_longest_read = 7;
    const uint64_t catcaag_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ** AAGCTA: AAG(0f), AGC(2f), CTA(3f)
    std::vector<representation_t> aagcta_representations;
    std::vector<read_id_t> aagcta_read_ids;
    std::vector<position_in_read_t> aagcta_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> aagcta_directions_of_reads;
    std::vector<representation_t> aagcta_unique_representations;
    std::vector<std::uint32_t> aagcta_first_occurrence_of_representations;

    // AAG(0f)
    aagcta_representations.push_back(0b000010);
    aagcta_read_ids.push_back(1);
    aagcta_positions_in_reads.push_back(0);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b000010);
    aagcta_first_occurrence_of_representations.push_back(0);
    // AGC(2r)
    aagcta_representations.push_back(0b001001);
    aagcta_read_ids.push_back(1);
    aagcta_positions_in_reads.push_back(2);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    aagcta_unique_representations.push_back(0b001001);
    aagcta_first_occurrence_of_representations.push_back(1);
    // CTA(3f)
    aagcta_representations.push_back(0b011100);
    aagcta_read_ids.push_back(1);
    aagcta_positions_in_reads.push_back(3);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b011100);
    aagcta_first_occurrence_of_representations.push_back(2);
    // trailing elements
    aagcta_first_occurrence_of_representations.push_back(3);

    const read_id_t aagcta_number_of_reads                              = 1;
    const read_id_t aagcta_smallest_read_id                             = 1;
    const read_id_t aagcta_largest_read_id                              = 1;
    const position_in_read_t aagcta_number_of_basepairs_in_longest_read = 6;
    const uint64_t aagcta_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ************* IndexCacheDevice tests *************

    IndexDescriptor catcaag_index_descriptor(0, 1);
    IndexDescriptor aagcta_index_descriptor(1, 1);
    std::vector<IndexDescriptor> catcaag_index_descriptors({catcaag_index_descriptor});
    std::vector<IndexDescriptor> aagcta_index_descriptors({aagcta_index_descriptor});
    std::vector<IndexDescriptor> catcaag_aagcta_index_descriptors({catcaag_index_descriptor, aagcta_index_descriptor});

    auto index_cache_host = std::make_shared<IndexCacheHost>(same_query_and_target,
                                                             allocator,
                                                             query_parser,
                                                             target_parser,
                                                             k,
                                                             w,
                                                             hash_representations,
                                                             filtering_parameter,
                                                             cuda_stream);

    IndexCacheDevice index_cache_device(same_query_and_target,
                                        index_cache_host);

    index_cache_host->generate_query_cache_content(catcaag_index_descriptors);
    ASSERT_ANY_THROW(index_cache_device.get_index_from_query_cache(catcaag_index_descriptor));
    index_cache_device.generate_query_cache_content(catcaag_index_descriptors);
    auto index_query_catcaag = index_cache_device.get_index_from_query_cache(catcaag_index_descriptor);
    check_if_index_is_correct(index_query_catcaag,
                              catcaag_representations,
                              catcaag_read_ids,
                              catcaag_positions_in_reads,
                              catcaag_directions_of_reads,
                              catcaag_unique_representations,
                              catcaag_first_occurrence_of_representations,
                              catcaag_number_of_reads,
                              catcaag_smallest_read_id,
                              catcaag_largest_read_id,
                              catcaag_number_of_basepairs_in_longest_read,
                              catcaag_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_device_same_query_and_target_1");
    ASSERT_ANY_THROW(index_cache_device.get_index_from_query_cache(aagcta_index_descriptor));
    ASSERT_ANY_THROW(index_cache_device.get_index_from_target_cache(catcaag_index_descriptor));
    ASSERT_ANY_THROW(index_cache_device.get_index_from_target_cache(aagcta_index_descriptor));

    index_cache_host->generate_target_cache_content(catcaag_aagcta_index_descriptors);
    ASSERT_ANY_THROW(index_cache_device.get_index_from_target_cache(catcaag_index_descriptor));
    ASSERT_ANY_THROW(index_cache_device.get_index_from_target_cache(aagcta_index_descriptor));
    index_cache_device.generate_target_cache_content(catcaag_aagcta_index_descriptors);

    auto index_target_catcaag = index_cache_device.get_index_from_target_cache(catcaag_index_descriptor);
    ASSERT_EQ(index_query_catcaag, index_target_catcaag); // check same object is used because same_query_and_target == true
    check_if_index_is_correct(index_target_catcaag,
                              catcaag_representations,
                              catcaag_read_ids,
                              catcaag_positions_in_reads,
                              catcaag_directions_of_reads,
                              catcaag_unique_representations,
                              catcaag_first_occurrence_of_representations,
                              catcaag_number_of_reads,
                              catcaag_smallest_read_id,
                              catcaag_largest_read_id,
                              catcaag_number_of_basepairs_in_longest_read,
                              catcaag_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_device_same_query_and_target_2");

    auto index_target_aagcta = index_cache_device.get_index_from_target_cache(aagcta_index_descriptor);
    check_if_index_is_correct(index_target_aagcta,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_device_same_query_and_target_3");

    // get the same query and target indices again and make sure they point to the same objects as the last time
    auto index_query_catcaag_1 = index_cache_device.get_index_from_query_cache(catcaag_index_descriptor);
    auto index_target_aagcta_1 = index_cache_device.get_index_from_target_cache(aagcta_index_descriptor);
    ASSERT_EQ(index_query_catcaag, index_query_catcaag_1);
    ASSERT_EQ(index_target_aagcta, index_target_aagcta_1);

    CGA_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    CGA_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

TEST(TestCudamapperIndexCaching, test_index_cache_device_not_the_same_query_and_target)
{
    // AAGCTA: AAG(0f), AGC(2r), CTA(3f)
    // CATCAAG: AAG(4f), ATC(1f), ATG(0r), CAA(3f)

    cudaStream_t cuda_stream;
    CGA_CU_CHECK_ERR(cudaStreamCreate(&cuda_stream));

    const bool same_query_and_target               = false;
    std::shared_ptr<io::FastaParser> query_parser  = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/aagcta.fasta");
    std::shared_ptr<io::FastaParser> target_parser = io::create_kseq_fasta_parser(std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag.fasta");
    DefaultDeviceAllocator allocator               = create_default_device_allocator();
    const std::uint64_t k                          = 3;
    const std::uint64_t w                          = 2;
    const bool hash_representations                = false;
    const double filtering_parameter               = 1.0;

    // ************* expected indices *************

    // ** AAGCTA: AAG(0f), AGC(2f), CTA(3f)
    std::vector<representation_t> aagcta_representations;
    std::vector<read_id_t> aagcta_read_ids;
    std::vector<position_in_read_t> aagcta_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> aagcta_directions_of_reads;
    std::vector<representation_t> aagcta_unique_representations;
    std::vector<std::uint32_t> aagcta_first_occurrence_of_representations;

    // AAG(0f)
    aagcta_representations.push_back(0b000010);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(0);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b000010);
    aagcta_first_occurrence_of_representations.push_back(0);
    // AGC(2r)
    aagcta_representations.push_back(0b001001);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(2);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    aagcta_unique_representations.push_back(0b001001);
    aagcta_first_occurrence_of_representations.push_back(1);
    // CTA(3f)
    aagcta_representations.push_back(0b011100);
    aagcta_read_ids.push_back(0);
    aagcta_positions_in_reads.push_back(3);
    aagcta_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    aagcta_unique_representations.push_back(0b011100);
    aagcta_first_occurrence_of_representations.push_back(2);
    // trailing elements
    aagcta_first_occurrence_of_representations.push_back(3);

    const read_id_t aagcta_number_of_reads                              = 1;
    const read_id_t aagcta_smallest_read_id                             = 0;
    const read_id_t aagcta_largest_read_id                              = 0;
    const position_in_read_t aagcta_number_of_basepairs_in_longest_read = 6;
    const uint64_t aagcta_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ** CATCAAG: AAG(4f), ATC(1f), CAA(3f), ATG(0r)
    std::vector<representation_t> catcaag_representations;
    std::vector<read_id_t> catcaag_read_ids;
    std::vector<position_in_read_t> catcaag_positions_in_reads;
    std::vector<SketchElement::DirectionOfRepresentation> catcaag_directions_of_reads;
    std::vector<representation_t> catcaag_unique_representations;
    std::vector<std::uint32_t> catcaag_first_occurrence_of_representations;

    // AAG(4f)
    catcaag_representations.push_back(0b000010);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(4);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b000010);
    catcaag_first_occurrence_of_representations.push_back(0);
    // ATC(1f)
    catcaag_representations.push_back(0b001101);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(1);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b001101);
    catcaag_first_occurrence_of_representations.push_back(1);
    // ATG(0r)
    catcaag_representations.push_back(0b001110);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(0);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    catcaag_unique_representations.push_back(0b001110);
    catcaag_first_occurrence_of_representations.push_back(2);
    // CAA(3f)
    catcaag_representations.push_back(0b010000);
    catcaag_read_ids.push_back(0);
    catcaag_positions_in_reads.push_back(3);
    catcaag_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    catcaag_unique_representations.push_back(0b010000);
    catcaag_first_occurrence_of_representations.push_back(3);
    // trailing elements
    catcaag_first_occurrence_of_representations.push_back(4);

    const read_id_t catcaag_number_of_reads                              = 1;
    const read_id_t catcaag_smallest_read_id                             = 0;
    const read_id_t catcaag_largest_read_id                              = 0;
    const position_in_read_t catcaag_number_of_basepairs_in_longest_read = 7;
    const uint64_t catcaag_maximum_kmer_size                             = sizeof(representation_t) * CHAR_BIT / 2;

    // ************* IndexCacheDevice tests *************

    // both descriptors are the same, but they are going to be used with different parsers
    IndexDescriptor index_descriptor(0, 1);
    std::vector<IndexDescriptor> index_descriptors({index_descriptor});

    auto index_cache_host = std::make_shared<IndexCacheHost>(same_query_and_target,
                                                             allocator,
                                                             query_parser,
                                                             target_parser,
                                                             k,
                                                             w,
                                                             hash_representations,
                                                             filtering_parameter,
                                                             cuda_stream);

    IndexCacheDevice index_cache_device(same_query_and_target,
                                        index_cache_host);

    index_cache_host->generate_query_cache_content(index_descriptors);
    ASSERT_ANY_THROW(index_cache_device.get_index_from_query_cache(index_descriptor));
    ASSERT_ANY_THROW(index_cache_device.get_index_from_target_cache(index_descriptor));

    index_cache_device.generate_query_cache_content(index_descriptors);
    auto index_query = index_cache_device.get_index_from_query_cache(index_descriptor);
    check_if_index_is_correct(index_query,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_device_not_the_same_query_and_target_1");
    ASSERT_ANY_THROW(index_cache_device.get_index_from_target_cache(index_descriptor));

    ASSERT_ANY_THROW(index_cache_device.generate_target_cache_content(index_descriptors));

    index_cache_host->generate_target_cache_content(index_descriptors);
    index_cache_device.generate_target_cache_content(index_descriptors);

    index_query       = index_cache_device.get_index_from_query_cache(index_descriptor);
    auto index_target = index_cache_device.get_index_from_target_cache(index_descriptor);
    ASSERT_NE(index_query, index_target);
    check_if_index_is_correct(index_query,
                              aagcta_representations,
                              aagcta_read_ids,
                              aagcta_positions_in_reads,
                              aagcta_directions_of_reads,
                              aagcta_unique_representations,
                              aagcta_first_occurrence_of_representations,
                              aagcta_number_of_reads,
                              aagcta_smallest_read_id,
                              aagcta_largest_read_id,
                              aagcta_number_of_basepairs_in_longest_read,
                              aagcta_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_device_not_the_same_query_and_target_2");
    check_if_index_is_correct(index_target,
                              catcaag_representations,
                              catcaag_read_ids,
                              catcaag_positions_in_reads,
                              catcaag_directions_of_reads,
                              catcaag_unique_representations,
                              catcaag_first_occurrence_of_representations,
                              catcaag_number_of_reads,
                              catcaag_smallest_read_id,
                              catcaag_largest_read_id,
                              catcaag_number_of_basepairs_in_longest_read,
                              catcaag_maximum_kmer_size,
                              cuda_stream,
                              "test_index_cache_device_not_the_same_query_and_target_3");

    // get the same query and target indices again and make sure they point to the same objects as the last time
    auto index_query_1  = index_cache_device.get_index_from_query_cache(index_descriptor);
    auto index_target_1 = index_cache_device.get_index_from_target_cache(index_descriptor);
    ASSERT_EQ(index_query, index_query_1);
    ASSERT_EQ(index_target, index_target_1);

    CGA_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream));
    CGA_CU_CHECK_ERR(cudaStreamDestroy(cuda_stream));
}

} // namespace cudamapper
} // namespace claragenomics
