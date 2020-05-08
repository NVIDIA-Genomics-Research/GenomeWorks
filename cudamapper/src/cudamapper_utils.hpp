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


std::vector<std::string> kmerize_string(const std::string& s, int kmer_length, int stride);

template<typename T>
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
float similarity(std::string a, std::string b, int kmer_length, int stride);

} // namespace cudamapper
} // namespace claragenomics
