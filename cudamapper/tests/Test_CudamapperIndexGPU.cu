/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cudamapper_file_location.hpp"
#include "../src/index_gpu.cuh"
#include "../src/minimizer.hpp"

#include <claragenomics/utils/mathutils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claragenomics
{
namespace cudamapper
{

namespace details
{
namespace index_gpu
{

// ************ Test find_first_occurrences_of_representations_kernel **************

void test_find_first_occurrences_of_representations_kernel(const thrust::host_vector<std::uint64_t>& representation_index_mask_h,
                                                           const thrust::host_vector<representation_t>& input_representations_h,
                                                           const thrust::host_vector<std::uint32_t>& expected_starting_index_of_each_representation_h,
                                                           const thrust::host_vector<representation_t>& expected_unique_representations_h,
                                                           const std::uint32_t number_of_threads)
{
    const thrust::device_vector<std::uint64_t> representation_index_mask_d(representation_index_mask_h);
    const thrust::device_vector<representation_t> input_representations_d(input_representations_h);
    ASSERT_EQ(expected_starting_index_of_each_representation_h.size(), representation_index_mask_h.back());
    ASSERT_EQ(expected_unique_representations_h.size(), representation_index_mask_h.back());

    const std::uint64_t number_of_unique_representations = representation_index_mask_h.back();
    ASSERT_EQ(expected_starting_index_of_each_representation_h.size(), number_of_unique_representations);
    ASSERT_EQ(expected_unique_representations_h.size(), number_of_unique_representations);
    thrust::device_vector<std::uint32_t> starting_index_of_each_representation_d(number_of_unique_representations);
    thrust::device_vector<representation_t> unique_representations_d(number_of_unique_representations);

    std::uint32_t number_of_blocks = (representation_index_mask_d.size() - 1) / number_of_threads + 1;

    find_first_occurrences_of_representations_kernel<<<number_of_blocks, number_of_threads>>>(representation_index_mask_d.data().get(),
                                                                                              input_representations_d.data().get(),
                                                                                              representation_index_mask_d.size(),
                                                                                              starting_index_of_each_representation_d.data().get(),
                                                                                              unique_representations_d.data().get());

    const thrust::host_vector<std::uint32_t> starting_index_of_each_representation_h(starting_index_of_each_representation_d);
    const thrust::host_vector<representation_t> unique_representations_h(unique_representations_d);

    ASSERT_EQ(starting_index_of_each_representation_h.size(), expected_starting_index_of_each_representation_h.size());
    ASSERT_EQ(unique_representations_h.size(), expected_unique_representations_h.size());
    for (std::size_t i = 0; i < expected_starting_index_of_each_representation_h.size(); ++i)
    {
        EXPECT_EQ(starting_index_of_each_representation_h[i], expected_starting_index_of_each_representation_h[i]) << "index: " << i;
        EXPECT_EQ(unique_representations_h[i], expected_unique_representations_h[i]) << "index: " << i;
    }
}

TEST(TestCudamapperIndexGPU, test_find_first_occurrences_of_representations_kernel_small_example)
{
    thrust::host_vector<std::uint64_t> representation_index_mask_h;
    thrust::host_vector<representation_t> input_representations_h;
    thrust::host_vector<std::uint32_t> expected_starting_index_of_each_representation_h;
    thrust::host_vector<representation_t> expected_unique_representations_h;
    representation_index_mask_h.push_back(1);
    input_representations_h.push_back(10);
    expected_starting_index_of_each_representation_h.push_back(0);
    expected_unique_representations_h.push_back(10);
    representation_index_mask_h.push_back(1);
    input_representations_h.push_back(10);
    representation_index_mask_h.push_back(1);
    input_representations_h.push_back(10);
    representation_index_mask_h.push_back(1);
    input_representations_h.push_back(10);
    //
    representation_index_mask_h.push_back(2);
    input_representations_h.push_back(20);
    expected_starting_index_of_each_representation_h.push_back(4);
    expected_unique_representations_h.push_back(20);
    //
    representation_index_mask_h.push_back(3);
    input_representations_h.push_back(30);
    expected_starting_index_of_each_representation_h.push_back(5);
    expected_unique_representations_h.push_back(30);
    representation_index_mask_h.push_back(3);
    input_representations_h.push_back(30);
    representation_index_mask_h.push_back(3);
    input_representations_h.push_back(30);
    representation_index_mask_h.push_back(3);
    input_representations_h.push_back(30);
    //
    representation_index_mask_h.push_back(4);
    input_representations_h.push_back(40);
    expected_starting_index_of_each_representation_h.push_back(9);
    expected_unique_representations_h.push_back(40);
    representation_index_mask_h.push_back(4);
    input_representations_h.push_back(40);
    representation_index_mask_h.push_back(4);
    input_representations_h.push_back(40);
    //
    representation_index_mask_h.push_back(5);
    input_representations_h.push_back(50);
    expected_starting_index_of_each_representation_h.push_back(12);
    expected_unique_representations_h.push_back(50);
    //
    representation_index_mask_h.push_back(6);
    input_representations_h.push_back(60);
    expected_starting_index_of_each_representation_h.push_back(13);
    expected_unique_representations_h.push_back(60);

    std::uint32_t number_of_threads = 3;

    test_find_first_occurrences_of_representations_kernel(representation_index_mask_h,
                                                          input_representations_h,
                                                          expected_starting_index_of_each_representation_h,
                                                          expected_unique_representations_h,
                                                          number_of_threads);
}

TEST(TestCudamapperIndexGPU, test_find_first_occurrences_of_representations_kernel_large_example)
{
    const std::uint64_t total_sketch_elements                    = 10000000;
    const std::uint32_t sketch_elements_with_same_representation = 1000;

    thrust::host_vector<std::uint64_t> representation_index_mask_h;
    thrust::host_vector<representation_t> input_representations_h;
    thrust::host_vector<std::size_t> expected_starting_index_of_each_representation_h;
    thrust::host_vector<representation_t> expected_unique_representations_h;
    for (std::size_t i = 0; i < total_sketch_elements; ++i)
    {
        representation_index_mask_h.push_back(i / sketch_elements_with_same_representation + 1);
        input_representations_h.push_back(representation_index_mask_h.back() * 10);
        if (i % sketch_elements_with_same_representation == 0)
        {
            expected_starting_index_of_each_representation_h.push_back(i);
            expected_unique_representations_h.push_back(input_representations_h.back());
        }
    }

    std::uint32_t number_of_threads = 256;

    test_find_first_occurrences_of_representations_kernel(representation_index_mask_h,
                                                          input_representations_h,
                                                          expected_starting_index_of_each_representation_h,
                                                          expected_unique_representations_h,
                                                          number_of_threads);
}

// ************ Test find_first_occurrences_of_representations **************

void test_find_first_occurrences_of_representations(const thrust::host_vector<representation_t>& representations_h,
                                                    const thrust::host_vector<std::uint32_t>& expected_starting_index_of_each_representation_h,
                                                    const thrust::host_vector<representation_t>& expected_unique_representations_h)
{
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CudaMallocAllocator>();
    device_buffer<representation_t> representations_d(representations_h.size(), allocator);
    cudautils::device_copy_n(representations_h.data(), representations_h.size(), representations_d.data()); // H2D

    device_buffer<std::uint32_t> starting_index_of_each_representation_d(allocator);
    device_buffer<representation_t> unique_representations_d(allocator);

    find_first_occurrences_of_representations(allocator,
                                              unique_representations_d,
                                              starting_index_of_each_representation_d,
                                              representations_d);

    thrust::host_vector<std::uint32_t> starting_index_of_each_representation_h(starting_index_of_each_representation_d.size());
    cudautils::device_copy_n(starting_index_of_each_representation_d.data(), starting_index_of_each_representation_d.size(), starting_index_of_each_representation_h.data()); // D2H
    thrust::host_vector<representation_t> unique_representations_h(unique_representations_d.size());
    cudautils::device_copy_n(unique_representations_d.data(), unique_representations_d.size(), unique_representations_h.data()); //D2H

    ASSERT_EQ(starting_index_of_each_representation_h.size(), expected_starting_index_of_each_representation_h.size());
    ASSERT_EQ(unique_representations_h.size(), expected_unique_representations_h.size());
    ASSERT_EQ(starting_index_of_each_representation_h.size(), unique_representations_h.size() + 1); // starting_index_of_each_representation_h has an additional element for the past-the-end element

    for (std::size_t i = 0; i < unique_representations_h.size(); ++i)
    {
        EXPECT_EQ(starting_index_of_each_representation_h[i], expected_starting_index_of_each_representation_h[i]) << "index: " << i;
        EXPECT_EQ(unique_representations_h[i], expected_unique_representations_h[i]) << "index: " << i;
    }
    EXPECT_EQ(starting_index_of_each_representation_h.back(), expected_starting_index_of_each_representation_h.back()) << "index: " << expected_starting_index_of_each_representation_h.size() - 1;
}

TEST(TestCudamapperIndexGPU, test_find_first_occurrences_of_representations_small_example)
{
    /// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    /// 0  0  0  0 12 12 12 12 12 12 23 23 23 32 32 32 32 32 46 46 46
    /// 1  0  0  0  1  0  0  0  0  0  1  0  0  1  0  0  0  0  1  0  0
    /// 1  1  1  1  2  2  2  2  2  2  3  3  3  4  4  4  4  4  5  5  5
    /// ^           ^                 ^        ^              ^       ^
    /// 0  4 10 13 18 21

    thrust::host_vector<representation_t> representations_h;
    thrust::host_vector<std::uint32_t> expected_starting_index_of_each_representation_h;
    thrust::host_vector<representation_t> expected_unique_representations_h;
    representations_h.push_back(0);
    expected_starting_index_of_each_representation_h.push_back(0);
    expected_unique_representations_h.push_back(0);
    representations_h.push_back(0);
    representations_h.push_back(0);
    representations_h.push_back(0);
    representations_h.push_back(12);
    expected_starting_index_of_each_representation_h.push_back(4);
    expected_unique_representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(12);
    representations_h.push_back(23);
    expected_starting_index_of_each_representation_h.push_back(10);
    expected_unique_representations_h.push_back(23);
    representations_h.push_back(23);
    representations_h.push_back(23);
    representations_h.push_back(32);
    expected_starting_index_of_each_representation_h.push_back(13);
    expected_unique_representations_h.push_back(32);
    representations_h.push_back(32);
    representations_h.push_back(32);
    representations_h.push_back(32);
    representations_h.push_back(32);
    representations_h.push_back(46);
    expected_starting_index_of_each_representation_h.push_back(18);
    expected_unique_representations_h.push_back(46);
    representations_h.push_back(46);
    representations_h.push_back(46);
    expected_starting_index_of_each_representation_h.push_back(21);

    test_find_first_occurrences_of_representations(representations_h,
                                                   expected_starting_index_of_each_representation_h,
                                                   expected_unique_representations_h);
}

TEST(TestCudamapperIndexGPU, test_find_first_occurrences_of_representations_large_example)
{
    const std::uint64_t total_sketch_elements                    = 10000000;
    const std::uint32_t sketch_elements_with_same_representation = 1000;

    thrust::host_vector<representation_t> representations_h;
    thrust::host_vector<std::uint32_t> expected_starting_index_of_each_representation_h;
    thrust::host_vector<representation_t> expected_unique_representations_h;

    for (std::size_t i = 0; i < total_sketch_elements; ++i)
    {
        representations_h.push_back(i / sketch_elements_with_same_representation);
        if (i % sketch_elements_with_same_representation == 0)
        {
            expected_starting_index_of_each_representation_h.push_back(i);
            expected_unique_representations_h.push_back(i / sketch_elements_with_same_representation);
        }
    }
    expected_starting_index_of_each_representation_h.push_back(total_sketch_elements);

    test_find_first_occurrences_of_representations(representations_h,
                                                   expected_starting_index_of_each_representation_h,
                                                   expected_unique_representations_h);
}

// ************ Test copy_rest_to_separate_arrays **************

template <typename ReadidPositionDirection, typename DirectionOfRepresentation>
void test_function_copy_rest_to_separate_arrays(const thrust::host_vector<ReadidPositionDirection>& rest_h,
                                                const thrust::host_vector<read_id_t>& expected_read_ids_h,
                                                const thrust::host_vector<position_in_read_t>& expected_positions_in_reads_h,
                                                const thrust::host_vector<DirectionOfRepresentation>& expected_directions_of_reads_h,
                                                const std::uint32_t threads)
{
    ASSERT_EQ(rest_h.size(), expected_read_ids_h.size());
    ASSERT_EQ(rest_h.size(), expected_positions_in_reads_h.size());
    ASSERT_EQ(rest_h.size(), expected_directions_of_reads_h.size());
    thrust::device_vector<read_id_t> generated_read_ids_d(rest_h.size());
    thrust::device_vector<position_in_read_t> generated_positions_in_reads_d(rest_h.size());
    thrust::device_vector<DirectionOfRepresentation> generated_directions_of_reads_d(rest_h.size());

    const thrust::device_vector<ReadidPositionDirection> rest_d(rest_h);

    const std::uint32_t blocks = ceiling_divide<int64_t>(rest_h.size(), threads);

    copy_rest_to_separate_arrays<<<blocks, threads>>>(rest_d.data().get(),
                                                      generated_read_ids_d.data().get(),
                                                      generated_positions_in_reads_d.data().get(),
                                                      generated_directions_of_reads_d.data().get(),
                                                      rest_h.size());

    const thrust::host_vector<read_id_t>& generated_read_ids_h(generated_read_ids_d);
    const thrust::host_vector<position_in_read_t>& generated_positions_in_reads_h(generated_positions_in_reads_d);
    const thrust::host_vector<DirectionOfRepresentation>& generated_directions_of_reads_h(generated_directions_of_reads_d);

    for (std::size_t i = 0; i < rest_h.size(); ++i)
    {
        EXPECT_EQ(generated_read_ids_h[i], expected_read_ids_h[i]);
        EXPECT_EQ(generated_positions_in_reads_h[i], expected_positions_in_reads_h[i]);
        EXPECT_EQ(generated_directions_of_reads_h[i], expected_directions_of_reads_h[i]);
    }
}

TEST(TestCudamapperIndexGPU, test_function_copy_rest_to_separate_arrays)
{
    thrust::host_vector<Minimizer::ReadidPositionDirection> rest_h;
    thrust::host_vector<read_id_t> expected_read_ids_h;
    thrust::host_vector<position_in_read_t> expected_positions_in_reads_h;
    thrust::host_vector<Minimizer::DirectionOfRepresentation> expected_directions_of_reads_h;

    rest_h.push_back({5, 8, 0});
    expected_read_ids_h.push_back(5);
    expected_positions_in_reads_h.push_back(8);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({15, 6, 0});
    expected_read_ids_h.push_back(15);
    expected_positions_in_reads_h.push_back(6);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({2, 4, 1});
    expected_read_ids_h.push_back(2);
    expected_positions_in_reads_h.push_back(4);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::REVERSE);
    rest_h.push_back({18, 15, 0});
    expected_read_ids_h.push_back(18);
    expected_positions_in_reads_h.push_back(15);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({6, 4, 1});
    expected_read_ids_h.push_back(6);
    expected_positions_in_reads_h.push_back(4);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::REVERSE);
    rest_h.push_back({6, 3, 1});
    expected_read_ids_h.push_back(6);
    expected_positions_in_reads_h.push_back(3);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::REVERSE);
    rest_h.push_back({89, 45, 0});
    expected_read_ids_h.push_back(89);
    expected_positions_in_reads_h.push_back(45);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({547, 25, 0});
    expected_read_ids_h.push_back(547);
    expected_positions_in_reads_h.push_back(25);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({14, 16, 1});
    expected_read_ids_h.push_back(14);
    expected_positions_in_reads_h.push_back(16);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::REVERSE);
    rest_h.push_back({18, 16, 0});
    expected_read_ids_h.push_back(18);
    expected_positions_in_reads_h.push_back(16);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({45, 44, 0});
    expected_read_ids_h.push_back(45);
    expected_positions_in_reads_h.push_back(44);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({65, 45, 1});
    expected_read_ids_h.push_back(65);
    expected_positions_in_reads_h.push_back(45);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::REVERSE);
    rest_h.push_back({15, 20, 0});
    expected_read_ids_h.push_back(15);
    expected_positions_in_reads_h.push_back(20);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({45, 654, 1});
    expected_read_ids_h.push_back(45);
    expected_positions_in_reads_h.push_back(654);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::REVERSE);
    rest_h.push_back({782, 216, 0});
    expected_read_ids_h.push_back(782);
    expected_positions_in_reads_h.push_back(216);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({255, 245, 1});
    expected_read_ids_h.push_back(255);
    expected_positions_in_reads_h.push_back(245);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::REVERSE);
    rest_h.push_back({346, 579, 0});
    expected_read_ids_h.push_back(346);
    expected_positions_in_reads_h.push_back(579);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({12, 8, 0});
    expected_read_ids_h.push_back(12);
    expected_positions_in_reads_h.push_back(8);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);
    rest_h.push_back({65, 42, 1});
    expected_read_ids_h.push_back(65);
    expected_positions_in_reads_h.push_back(42);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::REVERSE);
    rest_h.push_back({566, 42, 0});
    expected_read_ids_h.push_back(566);
    expected_positions_in_reads_h.push_back(42);
    expected_directions_of_reads_h.push_back(Minimizer::DirectionOfRepresentation::FORWARD);

    const std::uint32_t threads = 8;

    test_function_copy_rest_to_separate_arrays(rest_h,
                                               expected_read_ids_h,
                                               expected_positions_in_reads_h,
                                               expected_directions_of_reads_h,
                                               threads);
}

// ************ Test compress_unique_representations_after_filtering_kernel **************

void test_compress_unique_representations_after_filtering_kernel(const thrust::host_vector<representation_t>& unique_representations_before_compression_h,
                                                                 const thrust::host_vector<std::uint32_t>& first_occurrence_of_representation_before_compression_h,
                                                                 const thrust::host_vector<std::uint32_t>& new_unique_representation_index_h,
                                                                 const thrust::host_vector<representation_t>& expected_unique_representations_after_compression_h,
                                                                 const thrust::host_vector<std::uint32_t>& expected_first_occurrence_of_representation_after_compression_h,
                                                                 const std::int32_t number_of_threads)
{
    ASSERT_EQ(unique_representations_before_compression_h.size(), first_occurrence_of_representation_before_compression_h.size() - 1);
    ASSERT_EQ(first_occurrence_of_representation_before_compression_h.size(), new_unique_representation_index_h.size());
    const std::uint32_t number_of_unique_representations_after_compression = new_unique_representation_index_h.back(); // last element of new_unique_representation_index_h is equal to the number of unique representation after filtering
    ASSERT_EQ(number_of_unique_representations_after_compression, expected_unique_representations_after_compression_h.size());
    ASSERT_EQ(expected_unique_representations_after_compression_h.size(), expected_first_occurrence_of_representation_after_compression_h.size() - 1);

    const thrust::device_vector<representation_t> unique_representations_before_compression_d(unique_representations_before_compression_h);
    const thrust::device_vector<std::uint32_t> first_occurrence_of_representation_before_compression_d(first_occurrence_of_representation_before_compression_h);
    const thrust::device_vector<std::uint32_t> new_unique_representation_index_d(new_unique_representation_index_h);

    thrust::device_vector<representation_t> unique_representations_after_compression_d(expected_unique_representations_after_compression_h.size());
    thrust::device_vector<std::uint32_t> first_occurrence_of_representation_after_compression_d(expected_first_occurrence_of_representation_after_compression_h.size());

    const std::int32_t number_of_blocks = ceiling_divide<std::int64_t>(unique_representations_before_compression_h.size() + 1,
                                                                       number_of_threads);

    compress_unique_representations_after_filtering_kernel<<<number_of_blocks, number_of_threads>>>(unique_representations_before_compression_d.size(),
                                                                                                    unique_representations_before_compression_d.data().get(),
                                                                                                    first_occurrence_of_representation_before_compression_d.data().get(),
                                                                                                    new_unique_representation_index_d.data().get(),
                                                                                                    unique_representations_after_compression_d.data().get(),
                                                                                                    first_occurrence_of_representation_after_compression_d.data().get());

    const thrust::host_vector<representation_t> unique_representations_after_compression_h(unique_representations_after_compression_d);
    const thrust::host_vector<std::uint32_t> first_occurrence_of_representation_after_compression_h(first_occurrence_of_representation_after_compression_d);

    ASSERT_EQ(unique_representations_after_compression_h.size(), first_occurrence_of_representation_after_compression_h.size() - 1);
    for (std::int32_t i = 0; i < get_size(unique_representations_after_compression_h); ++i)
    {
        EXPECT_EQ(unique_representations_after_compression_h[i], expected_unique_representations_after_compression_h[i]) << "index: " << i;
        EXPECT_EQ(first_occurrence_of_representation_after_compression_h[i], expected_first_occurrence_of_representation_after_compression_h[i]) << "index: " << i;
    }
    // first_occurrence_of_representation_after_compression_h has one more element
    EXPECT_EQ(first_occurrence_of_representation_after_compression_h.back(), expected_first_occurrence_of_representation_after_compression_h.back());
}

TEST(TestCudamapperIndexGPU, test_compress_unique_representations_after_filtering_kernel_small_example)
{
    thrust::host_vector<representation_t> unique_representations_before_compression_h;
    thrust::host_vector<std::uint32_t> first_occurrence_of_representation_before_compression_h;
    thrust::host_vector<std::uint32_t> new_unique_representation_index_h;
    thrust::host_vector<representation_t> expected_unique_representations_after_compression_h;
    thrust::host_vector<std::uint32_t> expected_first_occurrence_of_representation_after_compression_h;

    // 4 <- filtering_threshold
    // 1  3  5  6  7    <- unique_representations_before_compression_h
    // 2  2  4  6  3  0 <- number_of_sketch_elements_with_representation_d (before filtering)
    // 2  2  0  0  3  0 <- number_of_sketch_elements_with_representation_h (after filtering)
    // 0  2  4  4  4  7 <- first_occurrence_of_representation_before_compression_h
    // 1  1  0  0  1    <- keep_representation_mask
    // 0  1  2  2  2  3 <- new_unique_representation_index_h
    //
    // 1 3 7   <- unique_representations_after_compression_h
    // 0 2 4 7 <- first_occurrence_of_representation_after_compression_h

    unique_representations_before_compression_h.push_back(1);
    unique_representations_before_compression_h.push_back(3);
    unique_representations_before_compression_h.push_back(5);
    unique_representations_before_compression_h.push_back(6);
    unique_representations_before_compression_h.push_back(7);

    first_occurrence_of_representation_before_compression_h.push_back(0);
    first_occurrence_of_representation_before_compression_h.push_back(2);
    first_occurrence_of_representation_before_compression_h.push_back(4);
    first_occurrence_of_representation_before_compression_h.push_back(4);
    first_occurrence_of_representation_before_compression_h.push_back(4);
    first_occurrence_of_representation_before_compression_h.push_back(7);

    new_unique_representation_index_h.push_back(0);
    new_unique_representation_index_h.push_back(1);
    new_unique_representation_index_h.push_back(2);
    new_unique_representation_index_h.push_back(2);
    new_unique_representation_index_h.push_back(2);
    new_unique_representation_index_h.push_back(3);

    expected_unique_representations_after_compression_h.push_back(1);
    expected_unique_representations_after_compression_h.push_back(3);
    expected_unique_representations_after_compression_h.push_back(7);

    expected_first_occurrence_of_representation_after_compression_h.push_back(0);
    expected_first_occurrence_of_representation_after_compression_h.push_back(2);
    expected_first_occurrence_of_representation_after_compression_h.push_back(4);
    expected_first_occurrence_of_representation_after_compression_h.push_back(7);

    const std::int32_t number_of_threads = 4;

    test_compress_unique_representations_after_filtering_kernel(unique_representations_before_compression_h,
                                                                first_occurrence_of_representation_before_compression_h,
                                                                new_unique_representation_index_h,
                                                                expected_unique_representations_after_compression_h,
                                                                expected_first_occurrence_of_representation_after_compression_h,
                                                                number_of_threads);
}

TEST(TestCudamapperIndexGPU, test_compress_unique_representations_after_filtering_kernel_large_example)
{
    const std::uint64_t total_unique_representations = 10000000;

    thrust::host_vector<representation_t> unique_representations_before_compression_h;
    thrust::host_vector<std::uint32_t> first_occurrence_of_representation_before_compression_h;
    thrust::host_vector<std::uint32_t> new_unique_representation_index_h;
    thrust::host_vector<representation_t> expected_unique_representations_after_compression_h;
    thrust::host_vector<std::uint32_t> expected_first_occurrence_of_representation_after_compression_h;

    // 0  1  2  3  4  5  6  7  8  9 10 11    <- unique_representations_before_compression_h
    // 0  5  5  0  5  5  0  5  5  0  5  5    <- numer_of_occurrences_after_filtering
    // 0  0  5 10 10 15 20 20 25 30 30 35 40 <- first_occurrence_of_representation_before_compression_h
    // 0  1  1  0  1  1  0  1  1  0  1  1    <- keep_representation_mask
    // 0  0  1  2  2  3  4  4  5  6  6  7  8 <- new_unique_representation_index_h
    // after compression
    // 1  2  4  5  7  8 10 11    <- expected_unique_representations_after_compression_h
    // 0  5 10 15 20 25 30 35 40 <- expected_first_occurrence_of_representation_after_compression_h

    // 0  1  2  3  4  5  6  7  8  9 10 11    <- unique_representations_before_compression_h
    // 0  0  5 10 10 15 20 20 25 30 30 35 40 <- first_occurrence_of_representation_before_compression_h
    // 0  0  1  2  2  3  4  4  5  6  6  7  8 <- new_unique_representation_index_h
    //    1  2     4  5     7  8    10 11    <- expected_unique_representations_after_compression_h
    //    0  5    10 15    20 25    30 35 40 <- expected_first_occurrence_of_representation_after_compression_h

    // 0th iteration
    unique_representations_before_compression_h.push_back(0);
    first_occurrence_of_representation_before_compression_h.push_back(0);
    new_unique_representation_index_h.push_back(0);
    // 1st iteration
    unique_representations_before_compression_h.push_back(1);
    first_occurrence_of_representation_before_compression_h.push_back(0);
    new_unique_representation_index_h.push_back(0);
    expected_unique_representations_after_compression_h.push_back(1);
    expected_first_occurrence_of_representation_after_compression_h.push_back(0);
    for (std::uint32_t i = 2; i < total_unique_representations; ++i)
    {
        unique_representations_before_compression_h.push_back(i);

        if (i % 3 == 0)
        {
            first_occurrence_of_representation_before_compression_h.push_back(first_occurrence_of_representation_before_compression_h.back() + 5);
            new_unique_representation_index_h.push_back(new_unique_representation_index_h.back() + 1);
        }
        if (i % 3 == 1)
        {
            first_occurrence_of_representation_before_compression_h.push_back(first_occurrence_of_representation_before_compression_h.back());
            new_unique_representation_index_h.push_back(new_unique_representation_index_h.back());
            expected_unique_representations_after_compression_h.push_back(expected_unique_representations_after_compression_h.back() + 2);
            expected_first_occurrence_of_representation_after_compression_h.push_back(expected_first_occurrence_of_representation_after_compression_h.back() + 5);
        }
        if (i % 3 == 2)
        {
            first_occurrence_of_representation_before_compression_h.push_back(first_occurrence_of_representation_before_compression_h.back() + 5);
            new_unique_representation_index_h.push_back(new_unique_representation_index_h.back() + 1);
            expected_unique_representations_after_compression_h.push_back(expected_unique_representations_after_compression_h.back() + 1);
            expected_first_occurrence_of_representation_after_compression_h.push_back(expected_first_occurrence_of_representation_after_compression_h.back() + 5);
        }
    }
    /// 10000000 % 3 = 1
    first_occurrence_of_representation_before_compression_h.push_back(first_occurrence_of_representation_before_compression_h.back());
    new_unique_representation_index_h.push_back(new_unique_representation_index_h.back());
    expected_first_occurrence_of_representation_after_compression_h.push_back(expected_first_occurrence_of_representation_after_compression_h.back() + 5);

    const std::int32_t number_of_threads = 256;

    test_compress_unique_representations_after_filtering_kernel(unique_representations_before_compression_h,
                                                                first_occurrence_of_representation_before_compression_h,
                                                                new_unique_representation_index_h,
                                                                expected_unique_representations_after_compression_h,
                                                                expected_first_occurrence_of_representation_after_compression_h,
                                                                number_of_threads);
}

// ************ Test compress_data_arrays_after_filtering_kernel **************

template <typename DirectionOfRepresentation>
void test_compress_data_arrays_after_filtering_kernel(const thrust::host_vector<std::uint32_t>& number_of_sketch_elements_with_representation_before_compression_h,
                                                      const thrust::host_vector<std::uint32_t>& first_occurrence_of_representation_before_filtering_h,
                                                      const thrust::host_vector<std::uint32_t>& first_occurrence_of_representation_after_compression_h,
                                                      const thrust::host_vector<std::uint32_t>& unique_representation_index_after_compression_h,
                                                      const thrust::host_vector<representation_t>& representations_before_compression_h,
                                                      const thrust::host_vector<read_id_t>& read_ids_before_compression_h,
                                                      const thrust::host_vector<position_in_read_t>& positions_in_reads_before_compression_h,
                                                      const thrust::host_vector<DirectionOfRepresentation>& directions_of_representations_before_compression_h,
                                                      const thrust::host_vector<representation_t>& expected_representations_after_compression_h,
                                                      const thrust::host_vector<read_id_t>& expected_read_ids_after_compression_h,
                                                      const thrust::host_vector<position_in_read_t>& expected_positions_in_reads_after_compression_h,
                                                      const thrust::host_vector<DirectionOfRepresentation>& expected_directions_of_representations_after_compression_h,
                                                      const std::int32_t number_of_threads)
{
    ASSERT_EQ(number_of_sketch_elements_with_representation_before_compression_h.size(), first_occurrence_of_representation_before_filtering_h.size());
    ASSERT_EQ(number_of_sketch_elements_with_representation_before_compression_h.size(), unique_representation_index_after_compression_h.size());

    const std::size_t number_of_sketch_elements_before_compression = first_occurrence_of_representation_before_filtering_h.back();
    ASSERT_EQ(representations_before_compression_h.size(), number_of_sketch_elements_before_compression);
    ASSERT_EQ(read_ids_before_compression_h.size(), number_of_sketch_elements_before_compression);
    ASSERT_EQ(positions_in_reads_before_compression_h.size(), number_of_sketch_elements_before_compression);
    ASSERT_EQ(directions_of_representations_before_compression_h.size(), number_of_sketch_elements_before_compression);

    const std::size_t number_of_sketch_elements_after_compression = first_occurrence_of_representation_after_compression_h.back();
    ASSERT_EQ(expected_representations_after_compression_h.size(), number_of_sketch_elements_after_compression);
    ASSERT_EQ(expected_read_ids_after_compression_h.size(), number_of_sketch_elements_after_compression);
    ASSERT_EQ(expected_positions_in_reads_after_compression_h.size(), number_of_sketch_elements_after_compression);
    ASSERT_EQ(expected_directions_of_representations_after_compression_h.size(), number_of_sketch_elements_after_compression);

    const thrust::device_vector<std::uint32_t> number_of_sketch_elements_with_representation_before_compression_d(number_of_sketch_elements_with_representation_before_compression_h);
    const thrust::device_vector<std::uint32_t> first_occurrence_of_representation_before_compression_d(first_occurrence_of_representation_before_filtering_h);
    const thrust::device_vector<std::uint32_t> first_occurrence_of_representation_after_compression_d(first_occurrence_of_representation_after_compression_h);
    const thrust::device_vector<std::uint32_t> unique_representation_index_after_compression_d(unique_representation_index_after_compression_h);
    const thrust::device_vector<representation_t> representations_before_compression_d(representations_before_compression_h);
    const thrust::device_vector<read_id_t> read_ids_before_compression_d(read_ids_before_compression_h);
    const thrust::device_vector<position_in_read_t> positions_in_reads_before_compression_d(positions_in_reads_before_compression_h);
    const thrust::device_vector<DirectionOfRepresentation> directions_of_representations_before_compression_d(directions_of_representations_before_compression_h);
    thrust::device_vector<representation_t> representations_after_compression_d(expected_representations_after_compression_h.size());
    thrust::device_vector<read_id_t> read_ids_after_compression_d(expected_read_ids_after_compression_h.size());
    thrust::device_vector<position_in_read_t> positions_in_reads_after_compression_d(expected_positions_in_reads_after_compression_h.size());
    thrust::device_vector<DirectionOfRepresentation> directions_of_representations_after_compression_d(expected_directions_of_representations_after_compression_h.size());

    const std::uint32_t number_of_unique_representations = number_of_sketch_elements_with_representation_before_compression_h.size() - 1; // array has an additional element in the end

    // launch one block per unique representation before compression
    const std::int32_t number_of_blocks = number_of_unique_representations;

    compress_data_arrays_after_filtering_kernel<<<number_of_blocks, number_of_threads>>>(number_of_unique_representations,
                                                                                         number_of_sketch_elements_with_representation_before_compression_d.data().get(),
                                                                                         first_occurrence_of_representation_before_compression_d.data().get(),
                                                                                         first_occurrence_of_representation_after_compression_d.data().get(),
                                                                                         unique_representation_index_after_compression_d.data().get(),
                                                                                         representations_before_compression_d.data().get(),
                                                                                         read_ids_before_compression_d.data().get(),
                                                                                         positions_in_reads_before_compression_d.data().get(),
                                                                                         directions_of_representations_before_compression_d.data().get(),
                                                                                         representations_after_compression_d.data().get(),
                                                                                         read_ids_after_compression_d.data().get(),
                                                                                         positions_in_reads_after_compression_d.data().get(),
                                                                                         directions_of_representations_after_compression_d.data().get());

    thrust::host_vector<representation_t> representations_after_compression_h(representations_after_compression_d);
    thrust::host_vector<read_id_t> read_ids_after_compression_h(read_ids_after_compression_d);
    thrust::host_vector<position_in_read_t> positions_in_reads_after_compression_h(positions_in_reads_after_compression_d);
    thrust::host_vector<DirectionOfRepresentation> directions_of_representations_after_compression_h(directions_of_representations_after_compression_d);

    ASSERT_EQ(expected_representations_after_compression_h.size(), representations_after_compression_h.size());
    ASSERT_EQ(expected_read_ids_after_compression_h.size(), read_ids_after_compression_h.size());
    ASSERT_EQ(expected_positions_in_reads_after_compression_h.size(), positions_in_reads_after_compression_h.size());
    ASSERT_EQ(expected_directions_of_representations_after_compression_h.size(), directions_of_representations_after_compression_h.size());

    for (std::int32_t i = 0; i < get_size(expected_representations_after_compression_h); ++i)
    {
        EXPECT_EQ(expected_representations_after_compression_h[i], representations_after_compression_h[i]) << "index: " << i;
        EXPECT_EQ(expected_read_ids_after_compression_h[i], read_ids_after_compression_h[i]) << "index: " << i;
        EXPECT_EQ(expected_positions_in_reads_after_compression_h[i], positions_in_reads_after_compression_h[i]) << "index: " << i;
        EXPECT_EQ(expected_directions_of_representations_after_compression_h[i], directions_of_representations_after_compression_h[i]) << "index: " << i;
    }
}

TEST(TestCudamapperIndexGPU, test_compress_data_arrays_after_filtering_kernel_small_example)
{
    // 4 <- filtering_threshold
    // 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    // 1  1  3  3  5  5  5  5  6  6  6  6  6  6  7  7  7 <- representations_before_compression
    // 0  1  3  5  3  4  6  6  0  1  2  2  2  3  7  8  9 <- read_ids_before_compression
    // 0  0  1  1  4  5  8  9  3  6  7  8  9  5  4  7  3 <- positions_in_reads_before_compression
    // F  F  F  F  R  R  R  F  R  F  F  R  R  F  F  R  R <- directions_of_reads_before_compression
    // 1  3  5  6  7    <- unique_representations_before_compression
    // 2  2  4  6  3  0 <- number_of_sketch_elements_with_representation (before filtering)
    // 0  2  4  8 14 17 <- first_occurrence_of_representation_before_filtering
    // 2  2  0  0  3  0 <- number_of_sketch_elements_with_representation_before_compression (after filtering)
    // 0  2  4  4  4  7 <- first_occurrence_of_representation_before_compression (after filtering)
    // 0  2  4  7       <- first_occurrence_of_representation_after_compression
    // 1  1  0  0  1    <- keep_representation_mask
    // 0  1  2  2  2  3 <- unique_representation_index_after_compression (keep_representation_mask_d after exclusive sum)
    //
    // after compression gives:
    // 0  1  2  3  4  5  6  7
    // 1  1  3  3  7  7  7    <- representations_before_compression
    // 0  1  3  5  7  8  9    <- read_ids_before_compression
    // 0  0  1  1  4  7  3    <- positions_in_reads_before_compression
    // F  F  F  F  F  R  R    <- directions_of_reads_before_compression

    // 1  1  3  3  5  5  5  5  6  6  6  6  6  6  7  7  7 <- representations_before_compression
    std::vector<representation_t> representations_before_compression_std({1, 1, 3, 3, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7});
    thrust::host_vector<representation_t> representations_before_compression_h(begin(representations_before_compression_std),
                                                                               end(representations_before_compression_std));
    // 0  1  3  5  3  4  6  6  0  1  2  2  2  3  7  8  9 <- read_ids_before_compression
    std::vector<read_id_t> read_ids_before_compression_std({0, 1, 3, 5, 3, 4, 6, 6, 0, 1, 2, 2, 2, 3, 7, 8, 9});
    thrust::host_vector<read_id_t> read_ids_before_compression_h(begin(read_ids_before_compression_std),
                                                                 end(read_ids_before_compression_std));
    // 0  0  1  1  4  5  8  9  3  6  7  8  9  5  4  7  3 <- positions_in_reads_before_compression
    std::vector<position_in_read_t> positions_in_reads_before_compression_std({0, 0, 1, 1, 4, 5, 8, 9, 3, 6, 7, 8, 9, 5, 4, 7, 3});
    thrust::host_vector<position_in_read_t> positions_in_reads_before_compression_h(begin(positions_in_reads_before_compression_std),
                                                                                    end(positions_in_reads_before_compression_std));
    // F  F  F  F  R  R  R  F  R  F  F  R  R  F  F  R  R <- directions_of_reads_before_compression
    thrust::host_vector<SketchElement::DirectionOfRepresentation> directions_of_representations_before_compression_h;
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);

    // 1  1  3  3  7  7  7 <- representations_before_compression
    std::vector<representation_t> expected_representations_after_compression_std({1, 1, 3, 3, 7, 7, 7});
    thrust::host_vector<representation_t> expected_representations_after_compression_h(begin(expected_representations_after_compression_std),
                                                                                       end(expected_representations_after_compression_std));
    // 0  1  3  5  7  8  9 <- read_ids_before_compression
    std::vector<read_id_t> expected_read_ids_after_compression_std({0, 1, 3, 5, 7, 8, 9});
    thrust::host_vector<read_id_t> expected_read_ids_after_compression_h(begin(expected_read_ids_after_compression_std),
                                                                         end(expected_read_ids_after_compression_std));
    // 0  0  1  1  4  7  3 <- positions_in_reads_before_compression
    std::vector<position_in_read_t> expected_positions_in_reads_after_compression_std({0, 0, 1, 1, 4, 7, 3});
    thrust::host_vector<position_in_read_t> expected_positions_in_reads_after_compression_h(begin(expected_positions_in_reads_after_compression_std),
                                                                                            end(expected_positions_in_reads_after_compression_std));
    // F  F  F  F  F  R  R <- directions_of_reads_before_compression
    thrust::host_vector<SketchElement::DirectionOfRepresentation> expected_directions_of_representations_after_compression_h;
    expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);

    // 2  2  0  0  3  0 <- number_of_sketch_elements_with_representation_before_compression
    std::vector<std::uint32_t> number_of_sketch_elements_with_representation_before_compression_std({2, 2, 0, 0, 3, 0});
    thrust::host_vector<std::uint32_t> number_of_sketch_elements_with_representation_before_compression_h(begin(number_of_sketch_elements_with_representation_before_compression_std),
                                                                                                          end(number_of_sketch_elements_with_representation_before_compression_std));
    // 0  2  4  8 14 17 <- first_occurrence_of_representation_before_filtering_h
    std::vector<std::uint32_t> first_occurrence_of_representation_before_filtering_std({0, 2, 4, 8, 14, 17});
    thrust::host_vector<std::uint32_t> first_occurrence_of_representation_before_filtering_h(begin(first_occurrence_of_representation_before_filtering_std),
                                                                                             end(first_occurrence_of_representation_before_filtering_std));
    // 0  2  4  7       <- first_occurrence_of_representation_after_compression
    std::vector<std::uint32_t> first_occurrence_of_representation_after_compression_std({0, 2, 4, 7});
    thrust::host_vector<std::uint32_t> first_occurrence_of_representation_after_compression_h(begin(first_occurrence_of_representation_after_compression_std),
                                                                                              end(first_occurrence_of_representation_after_compression_std));
    // 0  1  2  2  2  3 <- unique_representation_index_after_compression
    std::vector<std::uint32_t> unique_representation_index_after_compression_std({0, 1, 2, 2, 2, 3});
    thrust::host_vector<std::uint32_t> unique_representation_index_after_compression_h(begin(unique_representation_index_after_compression_std),
                                                                                       end(unique_representation_index_after_compression_std));

    const std::int32_t number_of_threads = 4;

    test_compress_data_arrays_after_filtering_kernel(number_of_sketch_elements_with_representation_before_compression_h,
                                                     first_occurrence_of_representation_before_filtering_h,
                                                     first_occurrence_of_representation_after_compression_h,
                                                     unique_representation_index_after_compression_h,
                                                     representations_before_compression_h,
                                                     read_ids_before_compression_h,
                                                     positions_in_reads_before_compression_h,
                                                     directions_of_representations_before_compression_h,
                                                     expected_representations_after_compression_h,
                                                     expected_read_ids_after_compression_h,
                                                     expected_positions_in_reads_after_compression_h,
                                                     expected_directions_of_representations_after_compression_h,
                                                     number_of_threads);
}

TEST(TestCudamapperIndexGPU, test_compress_data_arrays_after_filtering_kernel_large_example)
{
    //  6 <- filtering_threshold
    //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15|16 17 18 19 20 21 22 23 24 25 26
    //  0  0  0  0  0  0  1  1  1  1  1  2  2  2  2  2| 3  3  3  3  3  3  4  4  4  4  4... <- representations_before_compression
    //  0  1  2  3  4  5  1  2  3  4  5  2  3  4  5  6| 0  1  2  3  4  5  1  2  3  4  5... <- read_ids_before_compression
    // 10 11 12 13 14 15 11 12 13 14 15 12 13 14 15 16|10 11 12 13 14 15 11 12 13 14 15... <- positions_in_reads_before_compression
    //  F  R  F  R  F  R  F  R  F  R  F  R  F  R  F  R| F  R  F  R  F  R  F  R  F  R  F... <- directions_of_reads_before_compression
    //
    //  0  1  2| 3  4  5| 6  7  8| 9 10 11..    <- unique_representations_before_compression
    //  6  5  5| 6  5  5| 6  5  5| 6  5  5...   <- number_of_occurrences_before_filtering
    //  0  6 11|16 22 27|32 38 43|48 54 59... X <- first_occurrence_of_representation_before_filtering (with aditional element)
    //  0  5  5| 0  5  5| 0  5  5| 0  5  5... X <- number_of_occurrences_after_filtering (with aditional element)
    //  0  0  5|10 10 15|20 20 25|30 30 35... X <- first_occurrence_of_representation_before_compression (with aditional element)
    //  0  1  1| 0  1  1| 0  1  1| 0  1  1      <- keep_representation_mask
    //  0  0  1| 2  2  3| 4  4  5| 6  6  7... X <- unique_representation_index_after_compression (with aditional element)
    //  0  5  5| 0  5  5| 0  5  5| 0  5  5... 0 <- number_of_sketch_elements_with_representation_before_compression (after filtering) (with additional element)
    //
    // after compression
    //  0  1  2  3  4  5  6  7  8  9|10 11 12 13 14 15 16 17 18 19|20 21
    //  1  1  1  1  1  2  2  2  2  2| 4  4  4  4  4  5  5  5  5  5| 7  7... <- expected_representations_after_compression
    //  1  2  3  4  5  2  3  4  5  6| 1  2  3  4  5  2  3  4  5  6| 1  2... <- expected_read_ids_after_compression
    // 11 12 13 14 15 12 13 14 15 16|11 12 13 14 15 12 13 14 15 16|11 12    <- expected_positions_in_reads_after_compression
    //  F  R  F  R  F  R  F  R  F  R| F  R  F  R  F  R  F  R  F  R| F  R    <- expected_directions_of_reads_before_compression
    //

    thrust::host_vector<std::uint32_t> number_of_sketch_elements_with_representation_before_compression_h;
    thrust::host_vector<std::uint32_t> first_occurrence_of_representation_before_filtering_h;
    thrust::host_vector<std::uint32_t> first_occurrence_of_representation_after_compression_h;
    thrust::host_vector<std::uint32_t> unique_representation_index_after_compression_h;
    thrust::host_vector<representation_t> representations_before_compression_h;
    thrust::host_vector<read_id_t> read_ids_before_compression_h;
    thrust::host_vector<position_in_read_t> positions_in_reads_before_compression_h;
    thrust::host_vector<SketchElement::DirectionOfRepresentation> directions_of_representations_before_compression_h;
    thrust::host_vector<representation_t> expected_representations_after_compression_h;
    thrust::host_vector<read_id_t> expected_read_ids_after_compression_h;
    thrust::host_vector<position_in_read_t> expected_positions_in_reads_after_compression_h;
    thrust::host_vector<SketchElement::DirectionOfRepresentation> expected_directions_of_representations_after_compression_h;

    const std::uint64_t total_sketch_elements = 50000000; // = 3125000 * 16

    representation_t current_representation                                    = 0;
    std::uint32_t current_first_occurrence_of_representation_before_filtering  = 0;
    std::uint32_t current_first_occurrence_of_representation_after_compression = 0;
    std::uint32_t current_unique_representation_index_after_compression        = 0;
    for (std::uint64_t i = 0; i < total_sketch_elements;)
    {
        // first 6 elements
        for (std::uint64_t j = 0; j < 6; ++j)
        {
            representations_before_compression_h.push_back(current_representation);
            read_ids_before_compression_h.push_back(j);
            positions_in_reads_before_compression_h.push_back(j + 10);
            if (0 == j / 2)
                directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
            else
                directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
        }
        ++current_representation;
        i += 6;
        number_of_sketch_elements_with_representation_before_compression_h.push_back(0); // all sketch elements would get filtered out
        first_occurrence_of_representation_before_filtering_h.push_back(current_first_occurrence_of_representation_before_filtering);
        current_first_occurrence_of_representation_before_filtering += 6;
        unique_representation_index_after_compression_h.push_back(current_unique_representation_index_after_compression);
        current_unique_representation_index_after_compression += 0;

        // next 5 elements
        for (std::uint64_t j = 0; j < 5; ++j)
        {
            representations_before_compression_h.push_back(current_representation);
            expected_representations_after_compression_h.push_back(current_representation);
            read_ids_before_compression_h.push_back(j + 1);
            expected_read_ids_after_compression_h.push_back(j + 1);
            positions_in_reads_before_compression_h.push_back(j + 11);
            expected_positions_in_reads_after_compression_h.push_back(j + 11);
            if (0 == j / 2)
            {
                directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
                expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
            }
            else
            {
                directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
                expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
            }
        }
        ++current_representation;
        i += 5;
        number_of_sketch_elements_with_representation_before_compression_h.push_back(5);
        first_occurrence_of_representation_before_filtering_h.push_back(current_first_occurrence_of_representation_before_filtering);
        current_first_occurrence_of_representation_before_filtering += 5;
        first_occurrence_of_representation_after_compression_h.push_back(current_first_occurrence_of_representation_after_compression);
        current_first_occurrence_of_representation_after_compression += 5;
        unique_representation_index_after_compression_h.push_back(current_unique_representation_index_after_compression);
        current_unique_representation_index_after_compression += 1;

        // next 5 elements
        for (std::uint64_t j = 0; j < 5; ++j)
        {
            representations_before_compression_h.push_back(current_representation);
            expected_representations_after_compression_h.push_back(current_representation);
            read_ids_before_compression_h.push_back(j + 2);
            expected_read_ids_after_compression_h.push_back(j + 2);
            positions_in_reads_before_compression_h.push_back(j + 12);
            expected_positions_in_reads_after_compression_h.push_back(j + 12);
            if (0 == j / 2)
            {
                directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
                expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
            }
            else
            {
                directions_of_representations_before_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
                expected_directions_of_representations_after_compression_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
            }
        }
        ++current_representation;
        i += 5;
        number_of_sketch_elements_with_representation_before_compression_h.push_back(5);
        first_occurrence_of_representation_before_filtering_h.push_back(current_first_occurrence_of_representation_before_filtering);
        current_first_occurrence_of_representation_before_filtering += 5;
        first_occurrence_of_representation_after_compression_h.push_back(current_first_occurrence_of_representation_after_compression);
        current_first_occurrence_of_representation_after_compression += 5;
        unique_representation_index_after_compression_h.push_back(current_unique_representation_index_after_compression);
        current_unique_representation_index_after_compression += 1;
    }
    number_of_sketch_elements_with_representation_before_compression_h.push_back(0);
    first_occurrence_of_representation_before_filtering_h.push_back(current_first_occurrence_of_representation_before_filtering);
    first_occurrence_of_representation_after_compression_h.push_back(current_first_occurrence_of_representation_after_compression);
    unique_representation_index_after_compression_h.push_back(current_unique_representation_index_after_compression);

    const std::int32_t number_of_threads = 3; // intentionally going with such small blocks

    test_compress_data_arrays_after_filtering_kernel(number_of_sketch_elements_with_representation_before_compression_h,
                                                     first_occurrence_of_representation_before_filtering_h,
                                                     first_occurrence_of_representation_after_compression_h,
                                                     unique_representation_index_after_compression_h,
                                                     representations_before_compression_h,
                                                     read_ids_before_compression_h,
                                                     positions_in_reads_before_compression_h,
                                                     directions_of_representations_before_compression_h,
                                                     expected_representations_after_compression_h,
                                                     expected_read_ids_after_compression_h,
                                                     expected_positions_in_reads_after_compression_h,
                                                     expected_directions_of_representations_after_compression_h,
                                                     number_of_threads);
}

// ************ Test filter_out_most_common_representations **************

template <typename DirectionOfRepresentation>
void test_filter_out_most_common_representations(const double filtering_parameter,
                                                 const thrust::host_vector<representation_t>& input_representations_h,
                                                 const thrust::host_vector<read_id_t>& input_read_ids_h,
                                                 const thrust::host_vector<position_in_read_t>& input_positions_in_reads_h,
                                                 const thrust::host_vector<DirectionOfRepresentation>& input_directions_of_representations_h,
                                                 const thrust::host_vector<representation_t>& input_unique_representations_h,
                                                 const thrust::host_vector<std::uint32_t>& input_first_occurrence_of_representations_h,
                                                 const thrust::host_vector<representation_t>& expected_output_representations_h,
                                                 const thrust::host_vector<read_id_t>& expected_output_read_ids_h,
                                                 const thrust::host_vector<position_in_read_t>& expected_output_positions_in_reads_h,
                                                 const thrust::host_vector<DirectionOfRepresentation>& expected_output_directions_of_representations_h,
                                                 const thrust::host_vector<representation_t>& expected_output_unique_representations_h,
                                                 const thrust::host_vector<std::uint32_t>& expected_output_first_occurrence_of_representations_h)
{
    ASSERT_GE(filtering_parameter, 0u);
    ASSERT_LE(filtering_parameter, 1'000'000'000ull);
    ASSERT_EQ(input_representations_h.size(), input_read_ids_h.size());
    ASSERT_EQ(input_representations_h.size(), input_positions_in_reads_h.size());
    ASSERT_EQ(input_representations_h.size(), input_directions_of_representations_h.size());
    ASSERT_EQ(input_unique_representations_h.size(), input_first_occurrence_of_representations_h.size() - 1);
    ASSERT_EQ(input_representations_h.size(), input_first_occurrence_of_representations_h.back());
    ASSERT_EQ(expected_output_representations_h.size(), expected_output_read_ids_h.size());
    ASSERT_EQ(expected_output_representations_h.size(), expected_output_positions_in_reads_h.size());
    ASSERT_EQ(expected_output_representations_h.size(), expected_output_directions_of_representations_h.size());
    ASSERT_EQ(expected_output_unique_representations_h.size(), expected_output_first_occurrence_of_representations_h.size() - 1);
    ASSERT_EQ(expected_output_representations_h.size(), expected_output_first_occurrence_of_representations_h.back());

    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CudaMallocAllocator>();

    device_buffer<representation_t> representations_d(input_representations_h.size(), allocator);
    cudautils::device_copy_n(input_representations_h.data(), input_representations_h.size(), representations_d.data()); // H2D
    device_buffer<read_id_t> read_ids_d(input_read_ids_h.size(), allocator);
    cudautils::device_copy_n(input_read_ids_h.data(), input_read_ids_h.size(), read_ids_d.data()); // H2D
    device_buffer<position_in_read_t> positions_in_reads_d(input_positions_in_reads_h.size(), allocator);
    cudautils::device_copy_n(input_positions_in_reads_h.data(), input_positions_in_reads_h.size(), positions_in_reads_d.data()); // H2D
    device_buffer<DirectionOfRepresentation> directions_of_representations_d(input_directions_of_representations_h.size(), allocator);
    cudautils::device_copy_n(input_directions_of_representations_h.data(), input_directions_of_representations_h.size(), directions_of_representations_d.data()); // H2D
    device_buffer<representation_t> unique_representations_d(input_unique_representations_h.size(), allocator);
    cudautils::device_copy_n(input_unique_representations_h.data(), input_unique_representations_h.size(), unique_representations_d.data()); // H2D
    device_buffer<std::uint32_t> first_occurrence_of_representations_d(input_first_occurrence_of_representations_h.size(), allocator);
    cudautils::device_copy_n(input_first_occurrence_of_representations_h.data(), input_first_occurrence_of_representations_h.size(), first_occurrence_of_representations_d.data()); // H2D

    filter_out_most_common_representations(allocator,
                                           filtering_parameter,
                                           representations_d,
                                           read_ids_d,
                                           positions_in_reads_d,
                                           directions_of_representations_d,
                                           unique_representations_d,
                                           first_occurrence_of_representations_d);

    thrust::host_vector<representation_t> output_representations_h(representations_d.size());
    cudautils::device_copy_n(representations_d.data(), representations_d.size(), output_representations_h.data()); //D2H
    thrust::host_vector<read_id_t> output_read_ids_h(read_ids_d.size());
    cudautils::device_copy_n(read_ids_d.data(), read_ids_d.size(), output_read_ids_h.data()); //D2H
    thrust::host_vector<position_in_read_t> output_positions_in_reads_h(positions_in_reads_d.size());
    cudautils::device_copy_n(positions_in_reads_d.data(), positions_in_reads_d.size(), output_positions_in_reads_h.data()); //D2H
    thrust::host_vector<DirectionOfRepresentation> output_directions_of_representations_h(directions_of_representations_d.size());
    cudautils::device_copy_n(directions_of_representations_d.data(), directions_of_representations_d.size(), output_directions_of_representations_h.data()); // D2H
    thrust::host_vector<representation_t> output_unique_representations_h(unique_representations_d.size());
    cudautils::device_copy_n(unique_representations_d.data(), unique_representations_d.size(), output_unique_representations_h.data()); // D2H
    thrust::host_vector<std::uint32_t> output_first_occurrence_of_representations_h(first_occurrence_of_representations_d.size());
    cudautils::device_copy_n(first_occurrence_of_representations_d.data(), first_occurrence_of_representations_d.size(), output_first_occurrence_of_representations_h.data()); // D2H

    ASSERT_EQ(expected_output_representations_h.size(), output_representations_h.size());
    ASSERT_EQ(expected_output_representations_h.size(), output_read_ids_h.size());
    ASSERT_EQ(expected_output_representations_h.size(), output_positions_in_reads_h.size());
    ASSERT_EQ(expected_output_representations_h.size(), output_directions_of_representations_h.size());

    for (std::int32_t i = 0; i < get_size(representations_d); ++i)
    {
        EXPECT_EQ(expected_output_representations_h[i], output_representations_h[i]) << "index: " << i;
        EXPECT_EQ(expected_output_read_ids_h[i], output_read_ids_h[i]) << "index: " << i;
        EXPECT_EQ(expected_output_positions_in_reads_h[i], output_positions_in_reads_h[i]) << "index: " << i;
        EXPECT_EQ(expected_output_directions_of_representations_h[i], output_directions_of_representations_h[i]) << "index: " << i;
    }

    ASSERT_EQ(expected_output_unique_representations_h, output_unique_representations_h);
    ASSERT_EQ(expected_output_first_occurrence_of_representations_h, output_first_occurrence_of_representations_h);
    ASSERT_EQ(output_unique_representations_h.size(), output_first_occurrence_of_representations_h.size() - 1);
    ASSERT_EQ(output_representations_h.size(), output_first_occurrence_of_representations_h.back());

    for (std::int32_t i = 0; i < get_size(output_unique_representations_h); ++i)
    {
        EXPECT_EQ(expected_output_unique_representations_h[i], output_unique_representations_h[i]) << "index: " << i;
        EXPECT_EQ(expected_output_first_occurrence_of_representations_h[i], output_first_occurrence_of_representations_h[i]) << "index: " << i;
    }
    ASSERT_EQ(expected_output_first_occurrence_of_representations_h.back(), output_first_occurrence_of_representations_h.back());
}

TEST(TestCudamapperIndexGPU, test_filter_out_most_common_representations_small_example)
{
    // For example this index initinally contains 20 sketch elements:
    // 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
    // 1  1  3  3  5  5  5  5  6  6  6  6  6  6  7  7  7  8  8  8 <- representations (before filtering)
    // 0  1  3  5  3  4  6  6  0  1  2  2  2  3  7  8  9  1  2  3 <- read_ids (before filtering)
    // 0  0  1  1  4  5  8  9  3  6  7  8  9  5  4  7  3  7  8  9 <- positions_in_reads (before filtering)
    // F  F  F  F  R  R  R  F  R  F  F  R  R  F  F  R  R  F  F  F <- directions_of_reads (before filtering)
    // 1  3  5  6  7  8    <- unique_representations (before filtering)
    // 0  2  4  8 14 17 20 <- first_occurrence_of_representations (before filtering)
    //
    // For filtering_parameter = 0.2:
    // sketch_elementes_with_that_representation/total_sketch_element >= filtering_parameter
    // sketch_elementes_with_that_representation/20 >= 0.2 <=>
    // sketch_elementes_with_that_representation >= 20 * 0.2 <=>
    // sketch_elementes_with_that_representation >= 4 <=>
    // sketch element with representations with 4 or more sketch elements will be removed
    //
    // In the example above that means that representations 5 and 6 will be removed and that the output would be:
    // 0  1  2  3  4  5  6  7  8  9
    // 1  1  3  3  7  7  7  8  8  8 <- representations (after filtering)
    // 0  1  3  5  7  8  9  1  2  3 <- read_ids (after filtering)
    // 0  0  1  1  4  7  3  7  8  9 <- positions_in_reads (after filtering)
    // F  F  F  F  F  R  R  F  F  F <- directions_of_reads (after filtering)
    // 1  3  7  8    <- unique_representations (after filtering)
    // 0  2  4  7 10 <- first_occurrence_of_representations (after filtering)

    const double filtering_parameter = 0.2;

    const std::vector<representation_t> input_representations_std({1, 1, 3, 3, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8});
    const thrust::host_vector<representation_t> input_representations_h(std::begin(input_representations_std), std::end(input_representations_std));

    const std::vector<read_id_t> input_read_ids_std({0, 1, 3, 5, 3, 4, 6, 6, 0, 1, 2, 2, 2, 3, 7, 8, 9, 1, 2, 3});
    const thrust::host_vector<read_id_t> input_read_ids_h(std::begin(input_read_ids_std), std::end(input_read_ids_std));

    const std::vector<position_in_read_t> input_positions_in_reads_std({0, 0, 1, 1, 4, 5, 8, 9, 3, 6, 7, 8, 9, 5, 4, 7, 3, 7, 8, 9});
    const thrust::host_vector<position_in_read_t> input_positions_in_reads_h(std::begin(input_positions_in_reads_std), std::end(input_positions_in_reads_std));

    const std::vector<SketchElement::DirectionOfRepresentation> input_directions_of_representations_std({SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                         SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                         SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                         SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                         SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                         SketchElement::DirectionOfRepresentation::FORWARD});
    const thrust::host_vector<SketchElement::DirectionOfRepresentation> input_directions_of_representations_h(std::begin(input_directions_of_representations_std), std::end(input_directions_of_representations_std));

    const std::vector<representation_t> input_unique_representations_std({1, 3, 5, 6, 7, 8});
    const thrust::host_vector<representation_t> input_unique_representations_h(std::begin(input_unique_representations_std), std::end(input_unique_representations_std));

    const std::vector<std::uint32_t> input_first_occurrence_of_representations_std({0, 2, 4, 8, 14, 17, 20});
    const thrust::host_vector<std::uint32_t> input_first_occurrence_of_representations_h(std::begin(input_first_occurrence_of_representations_std), std::end(input_first_occurrence_of_representations_std));

    const std::vector<representation_t> expected_output_representations_std({1, 1, 3, 3, 7, 7, 7, 8, 8, 8});
    const thrust::host_vector<representation_t> expected_output_representations_h(std::begin(expected_output_representations_std), std::end(expected_output_representations_std));

    const std::vector<read_id_t> expected_output_read_ids_std({0, 1, 3, 5, 7, 8, 9, 1, 2, 3});
    const thrust::host_vector<read_id_t> expected_output_read_ids_h(std::begin(expected_output_read_ids_std), std::end(expected_output_read_ids_std));

    const std::vector<position_in_read_t> expected_output_positions_in_reads_std({0, 0, 1, 1, 4, 7, 3, 7, 8, 9});
    const thrust::host_vector<position_in_read_t> expected_output_positions_in_reads_h(std::begin(expected_output_positions_in_reads_std), std::end(expected_output_positions_in_reads_std));

    const std::vector<SketchElement::DirectionOfRepresentation> expected_output_directions_of_representations_std({SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                                   SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                                   SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                                   SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                                   SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                                   SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                                   SketchElement::DirectionOfRepresentation::REVERSE,
                                                                                                                   SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                                   SketchElement::DirectionOfRepresentation::FORWARD,
                                                                                                                   SketchElement::DirectionOfRepresentation::FORWARD});
    const thrust::host_vector<SketchElement::DirectionOfRepresentation> expected_output_directions_of_representations_h(std::begin(expected_output_directions_of_representations_std), std::end(expected_output_directions_of_representations_std));

    const std::vector<representation_t> expected_output_unique_representations_std({1, 3, 7, 8});
    const thrust::host_vector<representation_t> expected_output_unique_representations_h(std::begin(expected_output_unique_representations_std), std::end(expected_output_unique_representations_std));

    const std::vector<std::uint32_t> expected_output_first_occurrence_of_representations_std({0, 2, 4, 7, 10});
    const thrust::host_vector<std::uint32_t> expected_output_first_occurrence_of_representations_h(std::begin(expected_output_first_occurrence_of_representations_std), std::end(expected_output_first_occurrence_of_representations_std));

    test_filter_out_most_common_representations(filtering_parameter,
                                                input_representations_h,
                                                input_read_ids_h,
                                                input_positions_in_reads_h,
                                                input_directions_of_representations_h,
                                                input_unique_representations_h,
                                                input_first_occurrence_of_representations_h,
                                                expected_output_representations_h,
                                                expected_output_read_ids_h,
                                                expected_output_positions_in_reads_h,
                                                expected_output_directions_of_representations_h,
                                                expected_output_unique_representations_h,
                                                expected_output_first_occurrence_of_representations_h);
}

TEST(TestCudamapperIndexGPU, test_filter_out_most_common_representations_large_example)
{
    // Total sketch elements = 50'000'000
    //
    // Wanted filtering_threshold = 6
    // 6 = 50'000'000 * filtering_parameter <=> filtering_parameter = 6 / 50'000'000 * <=>
    // filtering_parameter = 0.00000012
    //
    //  6 <- filtering_threshold
    //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15|16 17 18 19 20 21 22 23 24 25 26
    //  0  0  0  0  0  0  1  1  1  1  1  2  2  2  2  2| 3  3  3  3  3  3  4  4  4  4  4... <- representations_before_compression
    //  0  1  2  3  4  5  1  2  3  4  5  2  3  4  5  6| 0  1  2  3  4  5  1  2  3  4  5... <- read_ids_before_compression
    // 10 11 12 13 14 15 11 12 13 14 15 12 13 14 15 16|10 11 12 13 14 15 11 12 13 14 15... <- positions_in_reads_before_compression
    //  F  R  F  R  F  R  F  R  F  R  F  R  F  R  F  R| F  R  F  R  F  R  F  R  F  R  F... <- directions_of_reads_before_compression
    //  0  1  2| 3  4  5| 6  7  8| 9 10 11..    <- unique_representations_before_filtering
    //  0  6 11|16 22 27|32 38 43|48 54 59... X <- first_occurrence_of_representation_before_filtering (with aditional element)
    //
    // after compression
    //  0  1  2  3  4  5  6  7  8  9|10 11 12 13 14 15 16 17 18 19|20 21
    //  1  1  1  1  1  2  2  2  2  2| 4  4  4  4  4  5  5  5  5  5| 7  7... <- expected_representations_after_compression
    //  1  2  3  4  5  2  3  4  5  6| 1  2  3  4  5  2  3  4  5  6| 1  2... <- expected_read_ids_after_compression
    // 11 12 13 14 15 12 13 14 15 16|11 12 13 14 15 12 13 14 15 16|11 12    <- expected_positions_in_reads_after_compression
    //  F  R  F  R  F  R  F  R  F  R| F  R  F  R  F  R  F  R  F  R| F  R    <- expected_directions_of_reads_before_compression
    //  1  2| 4  5| 7  8|...   <- expected_unique_representations_after_filtering
    //  0  5|10 15|20 25|... X <- first_occurrence_of_representation_after_filtering (with aditional element)

    const std::uint64_t total_sketch_elements = 50000000; // = 3125000 * 16
    const double filtering_parameter          = 0.00000012;

    thrust::host_vector<representation_t> input_representations_h;
    thrust::host_vector<read_id_t> input_read_ids_h;
    thrust::host_vector<position_in_read_t> input_positions_in_reads_h;
    thrust::host_vector<SketchElement::DirectionOfRepresentation> input_directions_of_representations_h;
    thrust::host_vector<representation_t> input_unique_representations_h;
    thrust::host_vector<std::uint32_t> input_first_occurrence_of_representations_h;
    thrust::host_vector<representation_t> expected_output_representations_h;
    thrust::host_vector<read_id_t> expected_output_read_ids_h;
    thrust::host_vector<position_in_read_t> expected_output_positions_in_reads_h;
    thrust::host_vector<SketchElement::DirectionOfRepresentation> expected_output_directions_of_representations_h;
    thrust::host_vector<representation_t> expected_output_unique_representations_h;
    thrust::host_vector<std::uint32_t> expected_output_first_occurrence_of_representations_h;

    representation_t current_representation                                   = 0;
    std::uint32_t current_first_occurrence_of_representation_before_filtering = 0;
    std::uint32_t current_first_occurrence_of_representation_after_filtering  = 0;

    for (std::uint64_t i = 0; i < total_sketch_elements;) // add 16 elements per iteration
    {
        // first 6 elements
        for (std::uint64_t j = 0; j < 6; ++j)
        {
            input_representations_h.push_back(current_representation);
            input_read_ids_h.push_back(j);
            input_positions_in_reads_h.push_back(j + 10);
            if (0 == j / 2)
                input_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
            else
                input_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
        }
        input_unique_representations_h.push_back(current_representation);
        input_first_occurrence_of_representations_h.push_back(current_first_occurrence_of_representation_before_filtering);
        current_first_occurrence_of_representation_before_filtering += 6;
        ++current_representation;
        i += 6;

        // next 5 elements
        for (std::uint64_t j = 0; j < 5; ++j)
        {
            input_representations_h.push_back(current_representation);
            expected_output_representations_h.push_back(current_representation);
            input_read_ids_h.push_back(j + 1);
            expected_output_read_ids_h.push_back(j + 1);
            input_positions_in_reads_h.push_back(j + 11);
            expected_output_positions_in_reads_h.push_back(j + 11);
            if (0 == j / 2)
            {
                input_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
                expected_output_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
            }
            else
            {
                input_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
                expected_output_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
            }
        }
        input_unique_representations_h.push_back(current_representation);
        expected_output_unique_representations_h.push_back(current_representation);
        input_first_occurrence_of_representations_h.push_back(current_first_occurrence_of_representation_before_filtering);
        expected_output_first_occurrence_of_representations_h.push_back(current_first_occurrence_of_representation_after_filtering);
        current_first_occurrence_of_representation_before_filtering += 5;
        current_first_occurrence_of_representation_after_filtering += 5;
        ++current_representation;
        i += 5;

        // next 5 elements
        for (std::uint64_t j = 0; j < 5; ++j)
        {
            input_representations_h.push_back(current_representation);
            expected_output_representations_h.push_back(current_representation);
            input_read_ids_h.push_back(j + 2);
            expected_output_read_ids_h.push_back(j + 2);
            input_positions_in_reads_h.push_back(j + 12);
            expected_output_positions_in_reads_h.push_back(j + 12);
            if (0 == j / 2)
            {
                input_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
                expected_output_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
            }
            else
            {
                input_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
                expected_output_directions_of_representations_h.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
            }
        }
        input_unique_representations_h.push_back(current_representation);
        expected_output_unique_representations_h.push_back(current_representation);
        input_first_occurrence_of_representations_h.push_back(current_first_occurrence_of_representation_before_filtering);
        expected_output_first_occurrence_of_representations_h.push_back(current_first_occurrence_of_representation_after_filtering);
        current_first_occurrence_of_representation_before_filtering += 5;
        current_first_occurrence_of_representation_after_filtering += 5;
        ++current_representation;
        i += 5;
    }
    input_first_occurrence_of_representations_h.push_back(current_first_occurrence_of_representation_before_filtering);
    expected_output_first_occurrence_of_representations_h.push_back(current_first_occurrence_of_representation_after_filtering);

    test_filter_out_most_common_representations(filtering_parameter,
                                                input_representations_h,
                                                input_read_ids_h,
                                                input_positions_in_reads_h,
                                                input_directions_of_representations_h,
                                                input_unique_representations_h,
                                                input_first_occurrence_of_representations_h,
                                                expected_output_representations_h,
                                                expected_output_read_ids_h,
                                                expected_output_positions_in_reads_h,
                                                expected_output_directions_of_representations_h,
                                                expected_output_unique_representations_h,
                                                expected_output_first_occurrence_of_representations_h);
}

} // namespace index_gpu
} // namespace details

void test_function(const std::string& filename,
                   const read_id_t first_read_id,
                   const read_id_t past_the_last_read_id,
                   const read_id_t expected_smallest_read_id,
                   const read_id_t expected_largest_read_id,
                   const std::uint64_t kmer_size,
                   const std::uint64_t window_size,
                   const std::vector<representation_t>& expected_representations,
                   const std::vector<position_in_read_t>& expected_positions_in_reads,
                   const std::vector<read_id_t>& expected_read_ids,
                   const std::vector<SketchElement::DirectionOfRepresentation>& expected_directions_of_reads,
                   const std::vector<representation_t>& expected_unique_representations,
                   const std::vector<std::uint32_t>& expected_first_occurrence_of_representations,
                   const std::vector<std::string>& expected_read_id_to_read_name,
                   const std::vector<std::uint32_t>& expected_read_id_to_read_length,
                   const read_id_t expected_number_of_reads,
                   const position_in_read_t expected_number_of_basepairs_in_longest_read,
                   const double filtering_parameter = 1.0)
{
    std::unique_ptr<io::FastaParser> parser    = io::create_kseq_fasta_parser(filename);
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CudaMallocAllocator>();
    IndexGPU<Minimizer> index(allocator,
                              *parser,
                              first_read_id,
                              past_the_last_read_id,
                              kmer_size,
                              window_size,
                              false,
                              filtering_parameter);

    ASSERT_EQ(index.number_of_reads(), expected_number_of_reads);
    if (0 == expected_number_of_reads)
    {
        return;
    }

    ASSERT_EQ(index.smallest_read_id(), expected_smallest_read_id);
    ASSERT_EQ(index.largest_read_id(), expected_largest_read_id);

    ASSERT_EQ(expected_number_of_basepairs_in_longest_read, index.number_of_basepairs_in_longest_read());

    ASSERT_EQ(expected_number_of_reads, expected_read_id_to_read_name.size());
    ASSERT_EQ(expected_number_of_reads, expected_read_id_to_read_length.size());
    for (read_id_t read_id = first_read_id; read_id < past_the_last_read_id; ++read_id)
    {
        ASSERT_EQ(index.read_id_to_read_length(read_id), expected_read_id_to_read_length[read_id - first_read_id]) << "read_id: " << read_id;
        ASSERT_EQ(index.read_id_to_read_name(read_id), expected_read_id_to_read_name[read_id - first_read_id]) << "read_id: " << read_id;
    }

    // check arrays
    const device_buffer<representation_t>& representations_d                             = index.representations();
    const device_buffer<position_in_read_t>& positions_in_reads_d                        = index.positions_in_reads();
    const device_buffer<read_id_t>& read_ids_d                                           = index.read_ids();
    const device_buffer<SketchElement::DirectionOfRepresentation>& directions_of_reads_d = index.directions_of_reads();
    thrust::host_vector<representation_t> representations_h(representations_d.size());
    cudautils::device_copy_n(representations_d.data(), representations_d.size(), representations_h.data()); // D2H
    thrust::host_vector<position_in_read_t> positions_in_reads_h(positions_in_reads_d.size());
    cudautils::device_copy_n(positions_in_reads_d.data(), positions_in_reads_d.size(), positions_in_reads_h.data()); // D2H
    thrust::host_vector<read_id_t> read_ids_h(read_ids_d.size());
    cudautils::device_copy_n(read_ids_d.data(), read_ids_d.size(), read_ids_h.data()); // D2H
    thrust::host_vector<SketchElement::DirectionOfRepresentation> directions_of_reads_h(directions_of_reads_d.size());
    cudautils::device_copy_n(directions_of_reads_d.data(), directions_of_reads_d.size(), directions_of_reads_h.data()); // D2H
    ASSERT_EQ(representations_h.size(), expected_representations.size());
    ASSERT_EQ(positions_in_reads_h.size(), expected_positions_in_reads.size());
    ASSERT_EQ(read_ids_h.size(), expected_read_ids.size());
    ASSERT_EQ(directions_of_reads_h.size(), expected_directions_of_reads.size());
    ASSERT_EQ(representations_h.size(), positions_in_reads_h.size());
    ASSERT_EQ(positions_in_reads_h.size(), read_ids_h.size());
    ASSERT_EQ(read_ids_h.size(), directions_of_reads_h.size());
    for (std::size_t i = 0; i < expected_positions_in_reads.size(); ++i)
    {
        EXPECT_EQ(representations_h[i], expected_representations[i]) << "i: " << i;
        EXPECT_EQ(positions_in_reads_h[i], expected_positions_in_reads[i]) << "i: " << i;
        EXPECT_EQ(read_ids_h[i], expected_read_ids[i]) << "i: " << i;
        EXPECT_EQ(directions_of_reads_h[i], expected_directions_of_reads[i]) << "i: " << i;
    }

    const device_buffer<representation_t>& unique_representations_d           = index.unique_representations();
    const device_buffer<std::uint32_t>& first_occurrence_of_representations_d = index.first_occurrence_of_representations();
    thrust::host_vector<representation_t> unique_representations_h(unique_representations_d.size());
    cudautils::device_copy_n(unique_representations_d.data(), unique_representations_d.size(), unique_representations_h.data()); // D2H
    thrust::host_vector<std::uint32_t> first_occurrence_of_representations_h(first_occurrence_of_representations_d.size());
    cudautils::device_copy_n(first_occurrence_of_representations_d.data(), first_occurrence_of_representations_d.size(), first_occurrence_of_representations_h.data()); // D2H
    ASSERT_EQ(expected_unique_representations.size() + 1, expected_first_occurrence_of_representations.size());
    ASSERT_EQ(unique_representations_h.size(), expected_unique_representations.size());
    ASSERT_EQ(first_occurrence_of_representations_h.size(), expected_first_occurrence_of_representations.size());
    for (std::size_t i = 0; i < expected_unique_representations.size(); ++i)
    {
        EXPECT_EQ(expected_unique_representations[i], unique_representations_h[i]) << "index: " << i;
        EXPECT_EQ(expected_first_occurrence_of_representations[i], first_occurrence_of_representations_h[i]) << "index: " << i;
    }
    EXPECT_EQ(expected_first_occurrence_of_representations.back(), expected_representations.size());
}

TEST(TestCudamapperIndexGPU, GATT_4_1)
{
    // >read_0
    // GATT

    // GATT = 0b10001111
    // AATC = 0b00001101 <- minimizer

    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta";
    const std::uint64_t minimizer_size = 4;
    const std::uint64_t window_size    = 1;

    std::vector<std::string> expected_read_id_to_read_name;
    expected_read_id_to_read_name.push_back("read_0");

    std::vector<std::uint32_t> expected_read_id_to_read_length;
    expected_read_id_to_read_length.push_back(4);

    std::vector<representation_t> expected_representations;
    std::vector<position_in_read_t> expected_positions_in_reads;
    std::vector<read_id_t> expected_read_ids;
    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
    std::vector<representation_t> expected_unique_representations;
    std::vector<std::uint32_t> expected_first_occurrence_of_representations;

    expected_representations.push_back(0b1101);
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    expected_unique_representations.push_back(0b1101);
    expected_first_occurrence_of_representations.push_back(0);

    expected_first_occurrence_of_representations.push_back(1);

    const read_id_t expected_number_of_reads                              = 1;
    const read_id_t expected_smallest_read_id                             = 0;
    const read_id_t expected_largest_read_id                              = 0;
    const position_in_read_t expected_number_of_basepairs_in_longest_read = 4;

    test_function(filename,
                  0,
                  1,
                  expected_smallest_read_id,
                  expected_largest_read_id,
                  minimizer_size,
                  window_size,
                  expected_representations,
                  expected_positions_in_reads,
                  expected_read_ids,
                  expected_directions_of_reads,
                  expected_unique_representations,
                  expected_first_occurrence_of_representations,
                  expected_read_id_to_read_name,
                  expected_read_id_to_read_length,
                  expected_number_of_reads,
                  expected_number_of_basepairs_in_longest_read);
}

TEST(TestCudamapperIndexGPU, GATT_2_3)
{
    // >read_0
    // GATT

    // kmer representation: forward, reverse
    // GA: <20> 31
    // AT: <03> 03
    // TT:  33 <00>

    // front end minimizers: representation, position_in_read, direction, read_id
    // GA : 20 0 F 0
    // GAT: 03 1 F 0

    // central minimizers
    // GATT: 00 2 R 0

    // back end minimizers
    // ATT: 00 2 R 0
    // TT : 00 2 R 0

    // All minimizers: GA(0f), AT(1f), AA(2r)

    // (2r1) means position 2, reverse direction, read 1
    // (1,2) means array block start at element 1 and has 2 elements

    //              0        1        2
    // data arrays: GA(0f0), AT(1f0), AA(2r0)

    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/gatt.fasta";
    const std::uint64_t minimizer_size = 2;
    const std::uint64_t window_size    = 3;

    std::vector<std::string> expected_read_id_to_read_name;
    expected_read_id_to_read_name.push_back("read_0");

    std::vector<std::uint32_t> expected_read_id_to_read_length;
    expected_read_id_to_read_length.push_back(4);

    std::vector<representation_t> expected_representations;
    std::vector<position_in_read_t> expected_positions_in_reads;
    std::vector<read_id_t> expected_read_ids;
    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
    std::vector<representation_t> expected_unique_representations;
    std::vector<std::uint32_t> expected_first_occurrence_of_representations;

    expected_representations.push_back(0b0000); // AA(2r0)
    expected_positions_in_reads.push_back(2);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    expected_unique_representations.push_back(0b0000);
    expected_first_occurrence_of_representations.push_back(0);
    expected_representations.push_back(0b0011); // AT(1f0)
    expected_positions_in_reads.push_back(1);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0011);
    expected_first_occurrence_of_representations.push_back(1);
    expected_representations.push_back(0b1000); // GA(0f0)
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b1000);
    expected_first_occurrence_of_representations.push_back(2);

    expected_first_occurrence_of_representations.push_back(3);

    const read_id_t expected_number_of_reads                              = 1;
    const read_id_t expected_smallest_read_id                             = 0;
    const read_id_t expected_largest_read_id                              = 0;
    const position_in_read_t expected_number_of_basepairs_in_longest_read = 4;

    test_function(filename,
                  0,
                  1,
                  expected_smallest_read_id,
                  expected_largest_read_id,
                  minimizer_size,
                  window_size,
                  expected_representations,
                  expected_positions_in_reads,
                  expected_read_ids,
                  expected_directions_of_reads,
                  expected_unique_representations,
                  expected_first_occurrence_of_representations,
                  expected_read_id_to_read_name,
                  expected_read_id_to_read_length,
                  expected_number_of_reads,
                  expected_number_of_basepairs_in_longest_read);
}

TEST(TestCudamapperIndexGPU, CCCATACC_2_8)
{
    // *** Read is shorter than one full window, the result should be empty ***

    // >read_0
    // CCCATACC

    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/cccatacc.fasta";
    const std::uint64_t minimizer_size = 2;
    const std::uint64_t window_size    = 8;

    // all data arrays should be empty

    std::vector<std::string> expected_read_id_to_read_name;

    std::vector<std::uint32_t> expected_read_id_to_read_length;

    std::vector<representation_t> expected_representations;
    std::vector<position_in_read_t> expected_positions_in_reads;
    std::vector<read_id_t> expected_read_ids;
    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
    std::vector<representation_t> expected_unique_representations;
    std::vector<std::uint32_t> expected_first_occurrence_of_representations;

    const read_id_t expected_number_of_reads                              = 0;
    const read_id_t expected_smallest_read_id                             = 0;
    const read_id_t expected_largest_read_id                              = 0;
    const position_in_read_t expected_number_of_basepairs_in_longest_read = 0;

    test_function(filename,
                  0,
                  1,
                  expected_smallest_read_id,
                  expected_largest_read_id,
                  minimizer_size,
                  window_size,
                  expected_representations,
                  expected_positions_in_reads,
                  expected_read_ids,
                  expected_directions_of_reads,
                  expected_unique_representations,
                  expected_first_occurrence_of_representations,
                  expected_read_id_to_read_name,
                  expected_read_id_to_read_length,
                  expected_number_of_reads,
                  expected_number_of_basepairs_in_longest_read);
}

// TODO: Cover this case as well
//TEST(TestCudamapperIndexGPU, CATCAAG_AAGCTA_3_5)
//{
//    // *** One Read is shorter than one full window, the other is not ***
//
//    // >read_0
//    // CATCAAG
//    // >read_1
//    // AAGCTA
//
//    // ** CATCAAG **
//
//    // kmer representation: forward, reverse
//    // CAT:  103 <032>
//    // ATC: <031> 203
//    // TCA: <310> 320
//    // CAA: <100> 332
//    // AAG: <002> 133
//
//    // front end minimizers: representation, position_in_read, direction, read_id
//    // CAT   : 032 0 R 0
//    // CATC  : 031 1 F 0
//    // CATCA : 031 1 F 0
//    // CATCAA: 031 1 F 0
//
//    // central minimizers
//    // CATCAAG: 002 4 F 0
//
//    // back end minimizers
//    // ATCAAG: 002 4 F 0
//    // TCAAG : 002 4 F 0
//    // CAAG  : 002 4 F 0
//    // AAG   : 002 4 F 0
//
//    // ** AAGCTA **
//    // ** read does not fit one array **
//
//    // All minimizers: ATG(0r0), ATC(1f0), AAG(4f0)
//
//    // (2r1) means position 2, reverse direction, read 1
//    // (1,2) means array block start at element 1 and has 2 elements
//
//    //              0         1         2
//    // data arrays: AAG(4f0), ATC(1f0), ATG(0r0)
//
//    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta";
//    const std::uint64_t minimizer_size = 3;
//    const std::uint64_t window_size    = 5;
//
//    std::vector<std::string> expected_read_id_to_read_name;
//    expected_read_id_to_read_name.push_back("read_0");
//
//    std::vector<std::uint32_t> expected_read_id_to_read_length;
//    expected_read_id_to_read_length.push_back(7);
//
//    std::vector<representation_t> expected_representations;
//    std::vector<position_in_read_t> expected_positions_in_reads;
//    std::vector<read_id_t> expected_read_ids;
//    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
//    expected_representations.push_back(0b000010); // AAG(4f0)
//    expected_positions_in_reads.push_back(4);
//    expected_read_ids.push_back(0);
//    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
//    expected_representations.push_back(0b001101); // ATC(1f0)
//    expected_positions_in_reads.push_back(1);
//    expected_read_ids.push_back(0);
//    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
//    expected_representations.push_back(0b001110); // ATG(0r0)
//    expected_positions_in_reads.push_back(0);
//    expected_read_ids.push_back(0);
//    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
//
//    const read_id_t expected_number_of_reads                              = 1;
//    const position_in_read_t expected_number_of_basepairs_in_longest_read = 7;
//
//    test_function(filename,
//                  0,
//                  2,
//                  minimizer_size,
//                  window_size,
//                  expected_representations,
//                  expected_positions_in_reads,
//                  expected_read_ids,
//                  expected_directions_of_reads,
//                  expected_read_id_to_read_name,
//                  expected_read_id_to_read_length,
//                  expected_number_of_reads,
//                  expected_number_of_basepairs_in_longest_read); // <- only one read goes into index, the other is too short
//}

TEST(TestCudamapperIndexGPU, CCCATACC_3_5)
{
    // >read_0
    // CCCATACC

    // ** CCCATAC **

    // kmer representation: forward, reverse
    // CCC: <111> 222
    // CCA: <110> 322
    // CAT:  103 <032>
    // ATA: <030> 303
    // TAC:  301 <230>
    // ACC: <011> 223

    // front end minimizers: representation, position_in_read, direction
    // CCC   : 111 0 F
    // CCCA  : 110 1 F
    // CCCAT : 032 2 R
    // CCCATA: 030 3 F

    // central minimizers
    // CCCATAC: 030 3 F
    // CCATACC: 011 5 F

    // back end minimizers
    // CATACC: 011 5 F
    // ATACC : 011 5 F
    // TACC  : 011 5 F
    // ACC   : 011 5 F

    // All minimizers: CCC(0f), CCA(1f), ATG(2r), ATA(3f), ACC(5f)

    // (2r1) means position 2, reverse direction, read 1
    // (1,2) means array block start at element 1 and has 2 elements

    //              0         1         2
    // data arrays: ACC(5f0), ATA(3f0), ATG(2r0), CCA(1f0), CCC(0f0)

    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/cccatacc.fasta";
    const std::uint64_t minimizer_size = 3;
    const std::uint64_t window_size    = 5;

    std::vector<std::string> expected_read_id_to_read_name;
    expected_read_id_to_read_name.push_back("read_0");

    std::vector<std::uint32_t> expected_read_id_to_read_length;
    expected_read_id_to_read_length.push_back(8);

    std::vector<representation_t> expected_representations;
    std::vector<position_in_read_t> expected_positions_in_reads;
    std::vector<read_id_t> expected_read_ids;
    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
    std::vector<representation_t> expected_unique_representations;
    std::vector<std::uint32_t> expected_first_occurrence_of_representations;

    expected_representations.push_back(0b000101); // ACC(5f0)
    expected_positions_in_reads.push_back(5);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b000101);
    expected_first_occurrence_of_representations.push_back(0);
    expected_representations.push_back(0b001100); // ATA(3f0)
    expected_positions_in_reads.push_back(3);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b001100);
    expected_first_occurrence_of_representations.push_back(1);
    expected_representations.push_back(0b001110); // ATG(2r0)
    expected_positions_in_reads.push_back(2);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    expected_unique_representations.push_back(0b001110);
    expected_first_occurrence_of_representations.push_back(2);
    expected_representations.push_back(0b010100); // CCA(1f0)
    expected_positions_in_reads.push_back(1);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b010100);
    expected_first_occurrence_of_representations.push_back(3);
    expected_representations.push_back(0b010101); // CCC(0f0)
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b010101);
    expected_first_occurrence_of_representations.push_back(4);

    expected_first_occurrence_of_representations.push_back(5);

    const read_id_t expected_number_of_reads                              = 1;
    const read_id_t expected_smallest_read_id                             = 0;
    const read_id_t expected_largest_read_id                              = 0;
    const position_in_read_t expected_number_of_basepairs_in_longest_read = 8;

    test_function(filename,
                  0,
                  1,
                  expected_smallest_read_id,
                  expected_largest_read_id,
                  minimizer_size,
                  window_size,
                  expected_representations,
                  expected_positions_in_reads,
                  expected_read_ids,
                  expected_directions_of_reads,
                  expected_unique_representations,
                  expected_first_occurrence_of_representations,
                  expected_read_id_to_read_name,
                  expected_read_id_to_read_length,
                  expected_number_of_reads,
                  expected_number_of_basepairs_in_longest_read);
}

TEST(TestCudamapperIndexGPU, CATCAAG_AAGCTA_3_2)
{
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

    // All minimizers: ATC(1f), CAA(3f), AAG(4f), ATG(0r)

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

    // All minimizers: AAG(0f), AGC(1f), CTA(3f)

    // (2r1) means position 2, reverse direction, read 1
    // (1,2) means array block start at element 1 and has 2 elements

    //              0         1         2         3         4         5         6
    // data arrays: AAG(4f0), AAG(0f1), AGC(2r1), ATC(1f0), ATG(0r0), CAA(3f0), CTA(3f1)

    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/catcaag_aagcta.fasta";
    const std::uint64_t minimizer_size = 3;
    const std::uint64_t window_size    = 2;

    std::vector<std::string> expected_read_id_to_read_name;
    expected_read_id_to_read_name.push_back("read_0");
    expected_read_id_to_read_name.push_back("read_1");

    std::vector<std::uint32_t> expected_read_id_to_read_length;
    expected_read_id_to_read_length.push_back(7);
    expected_read_id_to_read_length.push_back(6);

    std::vector<representation_t> expected_representations;
    std::vector<position_in_read_t> expected_positions_in_reads;
    std::vector<read_id_t> expected_read_ids;
    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
    std::vector<representation_t> expected_unique_representations;
    std::vector<std::uint32_t> expected_first_occurrence_of_representations;

    expected_representations.push_back(0b000010); // AAG(4f0)
    expected_positions_in_reads.push_back(4);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0010);
    expected_first_occurrence_of_representations.push_back(0);
    expected_representations.push_back(0b000010); // AAG(0f1)
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b001001); // AGC(2r1)
    expected_positions_in_reads.push_back(2);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    expected_unique_representations.push_back(0b001001);
    expected_first_occurrence_of_representations.push_back(2);
    expected_representations.push_back(0b001101); // ATC(1f0)
    expected_positions_in_reads.push_back(1);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b001101);
    expected_first_occurrence_of_representations.push_back(3);
    expected_representations.push_back(0b001110); // ATG(0r0)
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    expected_unique_representations.push_back(0b001110);
    expected_first_occurrence_of_representations.push_back(4);
    expected_representations.push_back(0b010000); // CAA(3f0)
    expected_positions_in_reads.push_back(3);
    expected_read_ids.push_back(0);
    expected_unique_representations.push_back(0b010000);
    expected_first_occurrence_of_representations.push_back(5);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b011100); // CTA(3f1)
    expected_positions_in_reads.push_back(3);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b011100);
    expected_first_occurrence_of_representations.push_back(6);

    expected_first_occurrence_of_representations.push_back(7);

    const read_id_t expected_number_of_reads                              = 2;
    const read_id_t expected_smallest_read_id                             = 0;
    const read_id_t expected_largest_read_id                              = 1;
    const position_in_read_t expected_number_of_basepairs_in_longest_read = 7;

    test_function(filename,
                  0,
                  2,
                  expected_smallest_read_id,
                  expected_largest_read_id,
                  minimizer_size,
                  window_size,
                  expected_representations,
                  expected_positions_in_reads,
                  expected_read_ids,
                  expected_directions_of_reads,
                  expected_unique_representations,
                  expected_first_occurrence_of_representations,
                  expected_read_id_to_read_name,
                  expected_read_id_to_read_length,
                  expected_number_of_reads,
                  expected_number_of_basepairs_in_longest_read);
}

TEST(TestCudamapperIndexGPU, AAAACTGAA_GCCAAAG_2_3)
{
    // >read_0
    // AAAACTGAA
    // >read_1
    // GCCAAAG

    // ** AAAACTGAA **

    // kmer representation: forward, reverse
    // AA: <00> 33
    // AA: <00> 33
    // AA: <00> 33
    // AC: <01> 23
    // CT:  13 <02>
    // TG:  32 <10>
    // GA: <20> 31
    // AA: <00> 33

    // front end minimizers: representation, position_in_read, direction, read_id
    // AA : 00 0 F 0
    // AAA: 00 1 F 0

    // central minimizers
    // AAAA: 00 2 F 0
    // AAAC: 00 2 F 0
    // AACT: 00 2 F 0
    // ACTG: 01 3 F 0
    // CTGA: 02 4 R 0
    // TGAA: 00 7 F 0

    // back end minimizers
    // GAA: 00 7 F 0
    // AA : 00 7 F 0

    // All minimizers: AA(0f), AA(1f), AA(2f), AC(3f), AG(4r), AA (7f)

    // ** GCCAAAG **

    // kmer representation: forward, reverse
    // GC: <21> 21
    // CC: <11> 22
    // CA: <10> 32
    // AA: <00> 33
    // AA: <00> 33
    // AG: <03> 21

    // front end minimizers: representation, position_in_read, direction, read_id
    // GC : 21 0 F 0
    // GCC: 11 1 F 0

    // central minimizers
    // GCCA: 10 2 F 0
    // CCAA: 00 3 F 0
    // CAAA: 00 4 F 0
    // AAAG: 00 4 F 0

    // back end minimizers
    // AAG: 00 4 F 0
    // AG : 03 5 F 0

    // All minimizers: GC(0f), CC(1f), CA(2f), AA(3f), AA(4f), AG(5f)

    // (2r1) means position 2, reverse direction, read 1
    // (1,2) means array block start at element 1 and has 2 elements

    //              0        1        2        3        4        5        6        7        8        9        10       11
    // data arrays: AA(0f0), AA(1f0), AA(2f0), AA(7f0), AA(3f1), AA(4f1), AC(3f0), AG(4r0), AG(5f1), CA(2f1), CC(1f1), GC(0f1)

    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/aaaactgaa_gccaaag.fasta";
    const std::uint64_t minimizer_size = 2;
    const std::uint64_t window_size    = 3;

    std::vector<std::string> expected_read_id_to_read_name;
    expected_read_id_to_read_name.push_back("read_0");
    expected_read_id_to_read_name.push_back("read_1");

    std::vector<std::uint32_t> expected_read_id_to_read_length;
    expected_read_id_to_read_length.push_back(9);
    expected_read_id_to_read_length.push_back(7);

    std::vector<representation_t> expected_representations;
    std::vector<position_in_read_t> expected_positions_in_reads;
    std::vector<read_id_t> expected_read_ids;
    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
    std::vector<representation_t> expected_unique_representations;
    std::vector<std::uint32_t> expected_first_occurrence_of_representations;

    expected_representations.push_back(0b0000); // AA(0f0)
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0000);
    expected_first_occurrence_of_representations.push_back(0);
    expected_representations.push_back(0b0000); // AA(1f0)
    expected_positions_in_reads.push_back(1);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b0000); // AA(2f0)
    expected_positions_in_reads.push_back(2);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b0000); // AA(7f0)
    expected_positions_in_reads.push_back(7);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b0000); // AA(3f1)
    expected_positions_in_reads.push_back(3);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b0000); // AA(4f1)
    expected_positions_in_reads.push_back(4);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b0001); // AC(3f0)
    expected_positions_in_reads.push_back(3);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0001);
    expected_first_occurrence_of_representations.push_back(6);
    expected_representations.push_back(0b0010); // AG(4r0)
    expected_positions_in_reads.push_back(4);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    expected_unique_representations.push_back(0b0010);
    expected_first_occurrence_of_representations.push_back(7);
    expected_representations.push_back(0b0010); // AG(5f1)
    expected_positions_in_reads.push_back(5);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b0100); // CA(2f1)
    expected_positions_in_reads.push_back(2);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0100);
    expected_first_occurrence_of_representations.push_back(9);
    expected_representations.push_back(0b0101); // CC(1f1)
    expected_positions_in_reads.push_back(1);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0101);
    expected_first_occurrence_of_representations.push_back(10);
    expected_representations.push_back(0b1001); // GC(0f1)
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b1001);
    expected_first_occurrence_of_representations.push_back(11);

    expected_first_occurrence_of_representations.push_back(12);

    const read_id_t expected_number_of_reads                              = 2;
    const read_id_t expected_smallest_read_id                             = 0;
    const read_id_t expected_largest_read_id                              = 1;
    const position_in_read_t expected_number_of_basepairs_in_longest_read = 9;

    test_function(filename,
                  0,
                  2,
                  expected_smallest_read_id,
                  expected_largest_read_id,
                  minimizer_size,
                  window_size,
                  expected_representations,
                  expected_positions_in_reads,
                  expected_read_ids,
                  expected_directions_of_reads,
                  expected_unique_representations,
                  expected_first_occurrence_of_representations,
                  expected_read_id_to_read_name,
                  expected_read_id_to_read_length,
                  expected_number_of_reads,
                  expected_number_of_basepairs_in_longest_read);
}

TEST(TestCudamapperIndexGPU, AAAACTGAA_GCCAAAG_2_3_only_second_read_in_index)
{
    // >read_0
    // AAAACTGAA
    // >read_1
    // GCCAAAG

    // ** AAAACTGAA **
    // only second read goes into index

    // ** GCCAAAG **

    // kmer representation: forward, reverse
    // GC: <21> 21
    // CC: <11> 22
    // CA: <10> 32
    // AA: <00> 33
    // AA: <00> 33
    // AG: <03> 21

    // front end minimizers: representation, position_in_read, direction, read_id
    // GC : 21 0 F 0
    // GCC: 11 1 F 0

    // central minimizers
    // GCCA: 10 2 F 0
    // CCAA: 00 3 F 0
    // CAAA: 00 4 F 0
    // AAAG: 00 4 F 0

    // back end minimizers
    // AAG: 00 4 F 0
    // AG : 03 5 F 0

    // All minimizers: GC(0f), CC(1f), CA(2f), AA(3f), AA(4f), AG(5f)

    // (2r1) means position 2, reverse direction, read 1
    // (1,2) means array block start at element 1 and has 2 elements

    //              0        1        2        3        4        5
    // data arrays: AA(3f1), AA(4f1), AG(5f1), CA(2f1), CC(1f1), GC(0f1)

    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/aaaactgaa_gccaaag.fasta";
    const std::uint64_t minimizer_size = 2;
    const std::uint64_t window_size    = 3;

    // only take second read
    std::vector<std::string> expected_read_id_to_read_name;
    expected_read_id_to_read_name.push_back("read_1");

    std::vector<std::uint32_t> expected_read_id_to_read_length;
    expected_read_id_to_read_length.push_back(7);

    std::vector<representation_t> expected_representations;
    std::vector<position_in_read_t> expected_positions_in_reads;
    std::vector<read_id_t> expected_read_ids;
    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
    std::vector<representation_t> expected_unique_representations;
    std::vector<std::uint32_t> expected_first_occurrence_of_representations;

    expected_representations.push_back(0b0000); // AA(3f1)
    expected_positions_in_reads.push_back(3);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b00);
    expected_first_occurrence_of_representations.push_back(0);
    expected_representations.push_back(0b0000); // AA(4f1)
    expected_positions_in_reads.push_back(4);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b0010); // AG(5f1)
    expected_positions_in_reads.push_back(5);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0010);
    expected_first_occurrence_of_representations.push_back(2);
    expected_representations.push_back(0b0100); // CA(2f1)
    expected_positions_in_reads.push_back(2);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0100);
    expected_first_occurrence_of_representations.push_back(3);
    expected_representations.push_back(0b0101); // CC(1f1)
    expected_positions_in_reads.push_back(1);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0101);
    expected_first_occurrence_of_representations.push_back(4);
    expected_representations.push_back(0b1001); // GC(0f1)
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b1001);
    expected_first_occurrence_of_representations.push_back(5);

    expected_first_occurrence_of_representations.push_back(6);

    const read_id_t expected_number_of_reads                              = 1;
    const read_id_t expected_smallest_read_id                             = 1; // <- index is instructed to ignore first (0th) read
    const read_id_t expected_largest_read_id                              = 1;
    const position_in_read_t expected_number_of_basepairs_in_longest_read = 7;

    test_function(filename,
                  1, // <- only take second read
                  2,
                  expected_smallest_read_id,
                  expected_largest_read_id,
                  minimizer_size,
                  window_size,
                  expected_representations,
                  expected_positions_in_reads,
                  expected_read_ids,
                  expected_directions_of_reads,
                  expected_unique_representations,
                  expected_first_occurrence_of_representations,
                  expected_read_id_to_read_name,
                  expected_read_id_to_read_length,
                  expected_number_of_reads,
                  expected_number_of_basepairs_in_longest_read);
}

TEST(TestCudamapperIndexGPU, AAAACTGAA_GCCAAAG_2_3_filtering)
{
    // >read_0
    // AAAACTGAA
    // >read_1
    // GCCAAAG

    // ** AAAACTGAA **

    // kmer representation: forward, reverse
    // AA: <00> 33
    // AA: <00> 33
    // AA: <00> 33
    // AC: <01> 23
    // CT:  13 <02>
    // TG:  32 <10>
    // GA: <20> 31
    // AA: <00> 33

    // front end minimizers: representation, position_in_read, direction, read_id
    // AA : 00 0 F 0
    // AAA: 00 1 F 0

    // central minimizers
    // AAAA: 00 2 F 0
    // AAAC: 00 2 F 0
    // AACT: 00 2 F 0
    // ACTG: 01 3 F 0
    // CTGA: 02 4 R 0
    // TGAA: 00 7 F 0

    // back end minimizers
    // GAA: 00 7 F 0
    // AA : 00 7 F 0

    // All minimizers: AA(0f), AA(1f), AA(2f), AC(3f), AG(4r), AA (7f)

    // ** GCCAAAG **

    // kmer representation: forward, reverse
    // GC: <21> 21
    // CC: <11> 22
    // CA: <10> 32
    // AA: <00> 33
    // AA: <00> 33
    // AG: <03> 21

    // front end minimizers: representation, position_in_read, direction, read_id
    // GC : 21 0 F 0
    // GCC: 11 1 F 0

    // central minimizers
    // GCCA: 10 2 F 0
    // CCAA: 00 3 F 0
    // CAAA: 00 4 F 0
    // AAAG: 00 4 F 0

    // back end minimizers
    // AAG: 00 4 F 0
    // AG : 03 5 F 0

    // All minimizers: GC(0f), CC(1f), CA(2f), AA(3f), AA(4f), AG(5f)

    // (2r1) means position 2, reverse direction, read 1
    // (1,2) means array block start at element 1 and has 2 elements

    //              0        1        2        3        4        5        6        7        8        9        10       11
    // data arrays: AA(0f0), AA(1f0), AA(2f0), AA(7f0), AA(3f1), AA(4f1), AC(3f0), AG(4r0), AG(5f1), CA(2f1), CC(1f1), GC(0f1)

    // Total sketch elements: 12
    // Sketch elements per representation: AA = 6, AC = 1, AG = 2, CA = 1, CC = 1, GC = 1
    // filtering_parameter = 0.5 <=> filtering_threshold = 12 * 0.5 = 6
    // data arrays after filtering: AC(3f0), AG(4r0), AG(5f1), CA(2f1), CC(1f1), GC(0f1)

    const std::string filename         = std::string(CUDAMAPPER_BENCHMARK_DATA_DIR) + "/aaaactgaa_gccaaag.fasta";
    const std::uint64_t minimizer_size = 2;
    const std::uint64_t window_size    = 3;
    const double filtering_parameter   = 0.5;

    std::vector<std::string> expected_read_id_to_read_name;
    expected_read_id_to_read_name.push_back("read_0");
    expected_read_id_to_read_name.push_back("read_1");

    std::vector<std::uint32_t> expected_read_id_to_read_length;
    expected_read_id_to_read_length.push_back(9);
    expected_read_id_to_read_length.push_back(7);

    std::vector<representation_t> expected_representations;
    std::vector<position_in_read_t> expected_positions_in_reads;
    std::vector<read_id_t> expected_read_ids;
    std::vector<SketchElement::DirectionOfRepresentation> expected_directions_of_reads;
    std::vector<representation_t> expected_unique_representations;
    std::vector<std::uint32_t> expected_first_occurrence_of_representations;

    expected_representations.push_back(0b0001); // AC(3f0)
    expected_positions_in_reads.push_back(3);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0001);
    expected_first_occurrence_of_representations.push_back(0);
    expected_representations.push_back(0b0010); // AG(4r0)
    expected_positions_in_reads.push_back(4);
    expected_read_ids.push_back(0);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::REVERSE);
    expected_unique_representations.push_back(0b0010);
    expected_first_occurrence_of_representations.push_back(1);
    expected_representations.push_back(0b0010); // AG(5f1)
    expected_positions_in_reads.push_back(5);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_representations.push_back(0b0100); // CA(2f1)
    expected_positions_in_reads.push_back(2);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0100);
    expected_first_occurrence_of_representations.push_back(3);
    expected_representations.push_back(0b0101); // CC(1f1)
    expected_positions_in_reads.push_back(1);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b0101);
    expected_first_occurrence_of_representations.push_back(4);
    expected_representations.push_back(0b1001); // GC(0f1)
    expected_positions_in_reads.push_back(0);
    expected_read_ids.push_back(1);
    expected_directions_of_reads.push_back(SketchElement::DirectionOfRepresentation::FORWARD);
    expected_unique_representations.push_back(0b1001);
    expected_first_occurrence_of_representations.push_back(5);

    expected_first_occurrence_of_representations.push_back(6);

    const read_id_t expected_number_of_reads                              = 2;
    const read_id_t expected_smallest_read_id                             = 0;
    const read_id_t expected_largest_read_id                              = 1;
    const position_in_read_t expected_number_of_basepairs_in_longest_read = 9;

    test_function(filename,
                  0,
                  2,
                  expected_smallest_read_id,
                  expected_largest_read_id,
                  minimizer_size,
                  window_size,
                  expected_representations,
                  expected_positions_in_reads,
                  expected_read_ids,
                  expected_directions_of_reads,
                  expected_unique_representations,
                  expected_first_occurrence_of_representations,
                  expected_read_id_to_read_name,
                  expected_read_id_to_read_length,
                  expected_number_of_reads,
                  expected_number_of_basepairs_in_longest_read,
                  filtering_parameter);
}

} // namespace cudamapper
} // namespace claragenomics
