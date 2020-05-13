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

#include <algorithm>
#include <vector>

#include <claragenomics/cudamapper/types.hpp>

namespace claragenomics
{
namespace cudamapper
{
/// \brief given a vector of overlaps, combines all overlaps from the same read pair
///
/// If two or more overlaps come from the same read pair they are combined into one large overlap:
/// Example:
/// Overlap 1:
///   Query ID = 18
///   Target ID = 42
///   Query start = 420
///   Query end = 520
///   Target start = 783
///   Target end = 883
/// Overlap 2:
///   Query ID = 18
///   Target ID = 42
///   Query start = 900
///   Query end = 1200
///   Target start = 1200
///   Target end = 1500
/// Fused overlap:
///   Query ID = 18
///   Target ID = 42
///   Query start = 420
///   Query end = 1200
///   Target start = 783
///   Target end = 1500
///
/// \param fused_overlaps Output vector for fused overlaps
/// \param unfused_overlaps vector of overlaps, sorted by (query_id, target_id) combination and query_start_position
void fuse_overlaps(std::vector<Overlap>& fused_overlaps, const std::vector<Overlap>& unfused_overlaps);

// ///@brief Given a string s, return the substring s[start, end].
// ///
// ///@param s A string
// ///@param start The 0-based start index of the returned substring.
// ///@param end The 0-based end index of the returned substring
// ///@return A std::string from s[start] to s[end].
// std::string string_slice(const std::string& s, std::size_t start, std::size_t end);

/// \brief Given a string s, produce its kmers (length <kmer-length>) and return them as a vector of strings.
/// \param s A string sequence to kmerize.
/// \param kmer_length A kmer length to use for producing kmers.
/// \param stride The number of bases to skip when selecting kmers (most often, this should be equal to 1).
/// \return A vector of strings containing the kmers (of length kmer_length) of s.
std::vector<std::string> split_into_kmers(const std::string& s, std::int32_t kmer_size, std::int32_t stride);

/// \brief Given two sorted vectors of comparable types, return a size_t count of the number of shared elements.
/// \param a A sorted vector of elements. These must be comparable (i.e., they must implement the == and < operators) and sorted in ascending order.
/// \param b A sorted vector of elements. These must be comparable with those in a and sorted in ascending order.
/// \return The number of elements the two sets have in common, including repeated elements, as a std::size_t.
template <typename T>
std::size_t count_shared_elements(const std::vector<T>& a, const std::vector<T>& b);

/// \brief Given two sequences 'a' and 'b', calculate an estimate of their similarity
/// Calculates the approximate nucleotide identity (or "similarity")
/// estimated from the Jaccard index of the kmers of strings a and b.
/// Note: This function assumes that a and b are on the same strand; you may need to
/// reverse-complement one of the strings if testing similarity on strings from different
/// strands.
/// \param a A C++ string
/// \param b A C++ string
/// \param kmer_length The kmer length to use for estimating similarity.
/// \param stride The number of bases to stride between kmers.
/// \return The estimated Jaccard index as a float.
float sequence_jaccard_similarity(const std::string& a, const std::string& b, std::int32_t kmer_size, std::int32_t stride);

/// \brief Given two sequences 'a' and 'b', calculate the Jaccard containment
/// The containment is similar to the Jaccard index (or similarity coefficient,
/// implemented in similarity) but is more robust to strings of different lengths.
/// Note: This function assumes that a and b are on the same strand; you may need to
/// reverse-complement one of the strings if testing similarity on strings from different
/// strands.
/// \param a A C++ string
/// \param b A C++ string
/// \param kmer_length The kmer length to use for estimating similarity.
/// \param stride The number of bases to stride between kmers.
/// \return The estimated Jaccard containment as a float.
float sequence_jaccard_containment(const std::string& a, const std::string& b, std::int32_t kmer_size, std::int32_t stride);

} // namespace cudamapper
} // namespace claragenomics
