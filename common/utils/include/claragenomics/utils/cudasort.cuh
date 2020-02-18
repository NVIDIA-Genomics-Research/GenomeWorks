/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <cub/cub.cuh>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/swap.h>

#include <claragenomics/utils/device_buffer.hpp>
#include <claragenomics/utils/mathutils.hpp>

namespace claragenomics
{
namespace cudautils
{
namespace details
{

/// \brief Sorts key-value pairs using radix sort
///
/// \param temp_storage_vect_d temporary storage, may be empty, function uses it if it is large enough, reallocates otherwise
/// \param unsorted_keys_d input
/// \param sorted_keys_d output
/// \param unsorted_values_d input
/// \param sorted_values_d output
/// \param number_of_elements number of elements to sort
/// \param begin_bit index of least significant bit to sort by (for example to sort numbers up to 253 = 0b1111'1101 this value should be 0)
/// \param end_bit index of past the most significant bit to sort by (for example to sort numbers up to 253 = 0b1111'1101 this value should be 8)
/// \tparam KeyT
/// \tparam ValueT
template <typename KeyT,
          typename ValueT>
void perform_radix_sort(device_buffer<char>& temp_storage_vect_d,
                        const KeyT* unsorted_keys_d,
                        KeyT* sorted_keys_d,
                        const ValueT* unsorted_values_d,
                        ValueT* sorted_values_d,
                        int number_of_elements,
                        std::uint32_t begin_bit,
                        std::uint32_t end_bit)
{
    // calculate necessary temp storage size
    void* temp_storage_d      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(temp_storage_d,
                                    temp_storage_bytes, // <- this one gets changed
                                    unsorted_keys_d,
                                    sorted_keys_d,
                                    unsorted_values_d,
                                    sorted_values_d,
                                    number_of_elements,
                                    begin_bit,
                                    end_bit);

    // allocate temp storage
    if (temp_storage_bytes > static_cast<size_t>(temp_storage_vect_d.size()))
    {
        // If directly calling resize new memory will be allocated before old is freed (beacause the data has to be copied from old to new memory)
        // For very large arrays this can lead to OOM, so manually deallocating old before allocating new memory
        temp_storage_vect_d.free();
        temp_storage_vect_d.resize(temp_storage_bytes);
    }
    temp_storage_d     = static_cast<void*>(temp_storage_vect_d.data());
    temp_storage_bytes = temp_storage_vect_d.size();

    // perform actual sort
    cub::DeviceRadixSort::SortPairs(temp_storage_d,
                                    temp_storage_bytes,
                                    unsorted_keys_d,
                                    sorted_keys_d,
                                    unsorted_values_d,
                                    sorted_values_d,
                                    number_of_elements,
                                    begin_bit,
                                    end_bit);
}

} // namespace details

/// \brief Sorts array by more significant key. Then sorts subarrays belonging to each more significiant key by less significant key
///
/// For example:
/// ms_key ls_key value
///      6      1     1
///      2      4     2
///      5      5     3
///      4      5     4
///      4      2     5
///      2      1     6
/// yields:
///      2      1     6
///      2      4     2
///      4      2     5
///      4      5     4
///      5      5     3
///      6      1     1
///
/// \param more_significant_keys sorted on output
/// \param less_significant_keys sorted on output
/// \param values sorted on output
/// \param max_value_of_more_significant_key optional, defaults to max value for MoreSignificantKeyT (specifying it might lead to better performance)
/// \param max_value_of_less_significant_key optional, defaults to max value for LessSignificantKeyT (specifying it might lead to better performance)
/// \tparam MoreSignificantKeyT
/// \tparam LessSignificantKeyT
/// \tparam ValueT
template <typename MoreSignificantKeyT,
          typename LessSignificantKeyT,
          typename ValueT>
void sort_by_two_keys(device_buffer<MoreSignificantKeyT>& more_significant_keys,
                      device_buffer<LessSignificantKeyT>& less_significant_keys,
                      device_buffer<ValueT>& values,
                      const MoreSignificantKeyT max_value_of_more_significant_key = std::numeric_limits<MoreSignificantKeyT>::max(),
                      const LessSignificantKeyT max_value_of_less_significant_key = std::numeric_limits<LessSignificantKeyT>::max())
{
    // Radix sort is done in-place, meaning sorting by less significant and then more significant keys yields the wanted result
    //
    // CUB's radix sort devides the key into chunks of a few bits (5-6?), sorts those bits and then moves to next (more significant) bits.
    // That way it has only a few dozens of buckets in every step.
    // A limitation of CUB's implementation is that it moves values to their new location in every step. If values are larger than 4 bytes this
    // begins to dominate. That's why it's better to use a 32-bit index and move the values to the final location in the end.

    // CUB accepts the number of elements as int, check if array is longer than that
    // (this limitation can be avoided by some CUB trickery, but no need to do it here)

    if (values.size() > std::numeric_limits<int>::max())
    {
        throw std::length_error("cudasort: array too long to be sorted");
    }
    using move_to_index_t = std::uint32_t;

    const auto number_of_elements = values.size();

    // TODO: Using CudaMallocAllocator for now. Switch to using the allocator used by input arrays
    //       once device_buffer::get_allocator() is added
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CudaMallocAllocator>();

    device_buffer<move_to_index_t> move_to_index(number_of_elements, allocator);
    // Fill array with values 0..number_of_elements-1
    thrust::sequence(thrust::device,
                     std::begin(move_to_index),
                     std::end(move_to_index));

    // *** sort by less significant key first ***
    device_buffer<LessSignificantKeyT> less_significant_key_sorted(number_of_elements, allocator);
    device_buffer<move_to_index_t> move_to_index_sorted(number_of_elements, allocator);

    device_buffer<char> temp_storage_vect(0, allocator);

    details::perform_radix_sort(temp_storage_vect,
                                less_significant_keys.data(),
                                less_significant_key_sorted.data(),
                                move_to_index.data(),
                                move_to_index_sorted.data(),
                                number_of_elements,
                                0,
                                int_floor_log2(max_value_of_less_significant_key) + 1); // function asks for the index of past-the-last most significant bit

    // swap sorted and unsorted arrays
    swap(less_significant_keys, less_significant_key_sorted);
    swap(move_to_index, move_to_index_sorted);

    // deallocate helper array
    // TODO: This array can probably be reused, but waiting for more general reallocation-avoidance strategy before optimizing this
    less_significant_key_sorted.free();

    // *** move more significant keys to their position after less significant keys sort ***
    device_buffer<MoreSignificantKeyT> more_significant_keys_after_sort(number_of_elements, allocator);

    thrust::gather(thrust::device,
                   std::begin(move_to_index),
                   std::end(move_to_index),
                   std::begin(more_significant_keys),
                   std::begin(more_significant_keys_after_sort));

    swap(more_significant_keys, more_significant_keys_after_sort);

    // *** sort by more significant key ***

    details::perform_radix_sort(temp_storage_vect,
                                more_significant_keys.data(),
                                more_significant_keys_after_sort.data(),
                                move_to_index.data(),
                                move_to_index_sorted.data(),
                                number_of_elements,
                                0,
                                int_floor_log2(max_value_of_more_significant_key) + 1);

    swap(move_to_index, move_to_index_sorted);

    // deallocate helper array
    move_to_index_sorted.free();

    // *** move the values to their final position ***
    device_buffer<ValueT> values_after_sort(number_of_elements, allocator);

    thrust::gather(thrust::device,
                   std::begin(move_to_index),
                   std::end(move_to_index),
                   std::begin(values),
                   std::begin(values_after_sort));

    swap(values, values_after_sort);
}

} // namespace cudautils
} // namespace claragenomics