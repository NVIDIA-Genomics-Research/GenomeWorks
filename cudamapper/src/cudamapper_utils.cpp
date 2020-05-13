/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
#include <vector>

#include "claragenomics/cudamapper/types.hpp"

namespace claragenomics
{
namespace cudamapper
{

void fuse_overlaps(std::vector<Overlap>& fused_overlaps, const std::vector<Overlap>& unfused_overlaps)
{
    // If the target start position is greater than the target end position
    // We can safely assume that the query and target are template and complement
    // reads. TODO: Incorporate sketchelement direction value when this is implemented
    auto set_relative_strand = [](Overlap& o) {
        if (o.target_start_position_in_read_ > o.target_end_position_in_read_)
        {
            o.relative_strand                = RelativeStrand::Reverse;
            auto tmp                         = o.target_end_position_in_read_;
            o.target_end_position_in_read_   = o.target_start_position_in_read_;
            o.target_start_position_in_read_ = tmp;
        }
        else
        {
            o.relative_strand = RelativeStrand::Forward;
        }
    };

    if (unfused_overlaps.size() == 0)
    {
        return;
    }

    Overlap fused_overlap = unfused_overlaps[0];

    for (size_t i = 0; i < unfused_overlaps.size() - 1; i++)
    {
        const Overlap& next_overlap = unfused_overlaps[i + 1];
        if ((fused_overlap.target_read_id_ == next_overlap.target_read_id_) &&
            (fused_overlap.query_read_id_ == next_overlap.query_read_id_))
        {
            //need to fuse
            fused_overlap.num_residues_ += next_overlap.num_residues_;
            fused_overlap.query_end_position_in_read_  = next_overlap.query_end_position_in_read_;
            fused_overlap.target_end_position_in_read_ = next_overlap.target_end_position_in_read_;
        }
        else
        {
            set_relative_strand(fused_overlap);
            fused_overlaps.push_back(fused_overlap);
            fused_overlap = unfused_overlaps[i + 1];
        }
    }

    set_relative_strand(fused_overlap);
    fused_overlaps.push_back(fused_overlap);
}

std::string string_slice(const std::string& s, const std::size_t start, const std::size_t end)
{
    return s.substr(start, end - start);
}

std::vector<std::string> split_into_kmers(const std::string& s, const std::int32_t kmer_size, const std::int32_t stride)
{
    std::size_t kmer_count = s.length() - kmer_size + 1;
    std::vector<std::string> kmers;

    if (s.length() < kmer_size)
    {
        return kmers;
    }

    for (std::size_t i = 0; i < kmer_count; i += stride)
    {
        kmers.push_back(s.substr(i, i + kmer_size));
    }
    return kmers;
}

template <typename T>
std::size_t count_shared_elements(const std::vector<T>& a, const std::vector<T>& b)
{
    std::size_t a_index      = 0;
    std::size_t b_index      = 0;
    std::size_t shared_count = 0;

    while (a_index < a.size() && b_index < b.size())
    {
        if (a[a_index] == b[b_index])
        {
            ++shared_count;
            ++a_index;
            ++b_index;
        }
        else if (a[a_index] < b[b_index])
        {
            ++a_index;
        }
        else
        {
            ++b_index;
        }
    }
    return shared_count;
}

float sequence_jaccard_containment(const std::string& a, const std::string& b, const std::int32_t kmer_size, const std::int32_t stride)
{
    std::vector<std::string> a_kmers = split_into_kmers(a, kmer_size, stride);
    std::vector<std::string> b_kmers = split_into_kmers(b, kmer_size, stride);
    std::sort(a_kmers.begin(), a_kmers.end());
    std::sort(b_kmers.begin(), b_kmers.end());

    std::size_t shared_kmers = count_shared_elements(a_kmers, b_kmers);
    // Calculate "containment", i.e., the total number of shared elements divided by
    // the number of elements in the smallest set. Min: 0, Max: 1.
    std::size_t shortest_kmer_set_length = std::min(a_kmers.size(), b_kmers.size());
    return static_cast<float>(shared_kmers) / static_cast<float>(shortest_kmer_set_length);
}

float sequence_jaccard_similarity(const std::string& a, const std::string& b, const std::int32_t kmer_size, const std::int32_t stride)
{
    std::vector<std::string> a_kmers = split_into_kmers(a, kmer_size, stride);
    std::vector<std::string> b_kmers = split_into_kmers(b, kmer_size, stride);
    std::sort(a_kmers.begin(), a_kmers.end());
    std::sort(b_kmers.begin(), b_kmers.end());

    std::size_t shared_kmers = count_shared_elements(a_kmers, b_kmers);
    // Calculate the set union size of a and b
    std::size_t union_size = a_kmers.size() + b_kmers.size() - shared_kmers;
    return static_cast<float>(shared_kmers) / static_cast<float>(union_size);
}

} // namespace cudamapper
} // namespace claragenomics
