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

#include <vector>
#include <thrust/device_vector.h>
#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/cudamapper/types.hpp>

namespace claragenomics
{

namespace cudamapper
{

class MatcherGPU
{
public:
    MatcherGPU(const Index& query_index,
               const Index& target_index);

    std::vector<Anchor>& anchors();

private:
    std::vector<Anchor> anchors_h_;
};

namespace details
{

namespace matcher_gpu
{
/// \brief Creates compressed representation of index
///
/// Creates an array in which n-th element represents the first occurrence of n-th representation.
/// Last element of the array is the total number of elements in representations_d array
///
/// For example:
/// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
/// 0  0  0  0 12 12 12 12 12 12 23 23 23 32 32 32 32 32 46 46 46
/// ^           ^                 ^        ^              ^       ^
/// gives:
/// 0 4 10 13 18 21
///
/// \param representations_d
/// \return first_element_for_representation
thrust::device_vector<std::uint32_t> find_first_occurrences_of_representations(const thrust::device_vector<representation_t>& representations_d);

/// \brief Finds the array index of the target representation for each query representation
///
/// Takes an array of query representations and an array of target representations
/// and checks for each query representation if the representation is present in the target array.
/// To return the result, the function takes a reference to an array of indices \param found_target_indices_d
/// which has to be of the same length of the query representations array.
/// If a query representation is found in the target the same representation the corresponding index
/// of the target array will be stored in \param found_target_indices_d at the position corresponding
/// to the query in the query array.
/// If a query is not found in the target array, -1 will be stored to the corresponding position of \param found_target_indices_d
/// For example:
///   query:
///     array-index:    0  1  2  3  4
///     representation: 0 12 23 32 46
///   target:
///     array-index:    0  1  2  3  4  5  6
///     representation: 5 12 16 23 24 25 46
///
/// gives:
///   found_target_indicies_d:
///     array-index:    0  1  2  3  4
///     target-index:  -1  1  3 -1  6
///
/// \param found_target_indices_d The array which will filled with the resulting target indices. This array has to be of same size as query_representations_d.
/// \param query_representations_d An array of query representations
/// \param target_representations_d An sorted array of target representations
void find_query_target_matches(thrust::device_vector<std::int64_t>& found_target_indices_d, const thrust::device_vector<representation_t>& query_representations_d, const thrust::device_vector<representation_t>& target_representations_d);

/// \brief Computes the number of anchors for matches in a query and target arrays.
///
/// Takes the arrays which store the positions of the first occurrences the different representations
/// in the query and target representation arrays (see find_first_occurrences_of_representations)
/// and the array with the found matches (see find_query_target_matches) and computes the total number
/// of anchors which can be computed from the query and target arrays.
/// The number of anchors is the number of all-to-all combinations of the matching representations in query and target.
/// For example:
///   query:
///     representation: 0 12 23 32 46
///     starting index: 0  4 10 13 18 21
///   target:
///     representation: 5 12 16 23 24 25 46
///     starting index: 0  3  7  9 13 16 18 21
///
///   found_target_indicies_d: (matching representations: 12, 23, 46)
///     array-index:    0  1  2  3  4
///     target-index:  -1  1  3 -1  6 (-1 indicates no matching representation in target)
///
///   gives:
///     (10-4)*(7-3) + (13-10)*(13-9) + (21-18)*(21-18) = 45
///
/// \param query_starting_index_of_each_representation_d
/// \param found_target_indices_d
/// \param target_starting_index_of_each_representation_d
/// \return The number of anchors which can be generated from the query and target arrays
std::int64_t compute_number_of_anchors(const thrust::device_vector<std::uint32_t> query_starting_index_of_each_representation_d, const thrust::device_vector<std::int64_t>& found_target_indices_d, const thrust::device_vector<std::uint32_t> target_starting_index_of_each_representation_d);

/// \brief Writes 0 to the output array if the value to the left is the same as the current value, 1 otherwise. First element is always 1
///
/// For example:
/// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
/// 0  0  0  0 12 12 12 12 12 12 23 23 23 32 32 32 32 32 46 46 46
/// gives:
/// 1  0  0  0  1  0  0  0  0  0  1  0  0  1  0  0  0  0  1  0  0
///
/// \param representations_d
/// \param number_of_elements
/// \param new_value_mask_d generated array
__global__ void create_new_value_mask(const representation_t* const representations_d,
                                      const std::size_t number_of_elements,
                                      std::uint32_t* const new_value_mask_d);

/// \brief Creates an array in which each element represents the index in representation_index_mask_d at which a new representation starts
///
/// For example:
/// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
/// 0  0  0  0 12 12 12 12 12 12 23 23 23 32 32 32 32 32 46 46 46
/// 1  0  0  0  1  0  0  0  0  0  1  0  0  1  0  0  0  0  1  0  0
/// 1  1  1  1  2  2  2  2  2  2  3  3  3  4  4  4  4  4  5  5  5
/// ^           ^                 ^        ^              ^
/// gives:
/// 0  4 10 13 18
///
/// \param representation_index_mask_d
/// \param number_of_input_elements
/// \param starting_index_of_each_representation
__global__ void copy_index_of_first_occurence(const std::uint64_t* const representation_index_mask_d,
                                              const std::size_t number_of_input_elements,
                                              std::uint32_t* const starting_index_of_each_representation);

/// \brief Performs a binary search on target_representations_d for each element of query_representations_d and stores the found index (or -1 iff not found) in found_target_indices.
///
/// For example:
///   query:
///     array-index:    0  1  2  3  4
///     representation: 0 12 23 32 46
///   target:
///     array-index:    0  1  2  3  4  5  6
///     representation: 5 12 16 23 24 25 46
///
/// gives:
///   found_target_indicies_d:
///     array-index:    0  1  2  3  4
///     target-index:  -1  1  3 -1  6
///
/// \param found_target_indices_d the array which will hold the result
/// \param query_representations_d the array of queries
/// \param n_query_representations size of \param query_representations_d and \param found_target_indices_d
/// \param target_representations_d the array of targets to be searched
/// \param n_target_representations size of \param target_representations_d
__global__ void find_query_target_matches_kernel(int64_t* const found_target_indices_d, const representation_t* const query_representations_d, const int64_t n_query_representations, const representation_t* const target_representations_d, const int64_t n_target_representations);
} // namespace matcher_gpu

} // namespace details

} // namespace cudamapper

} // namespace claragenomics
