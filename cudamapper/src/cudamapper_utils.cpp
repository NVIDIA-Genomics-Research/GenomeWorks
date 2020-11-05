/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "cudamapper_utils.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

std::vector<gw_string_view_t> split_into_kmers(const gw_string_view_t& s, const std::int32_t kmer_size, const std::int32_t stride)
{
    const std::size_t kmer_count = s.length() - kmer_size + 1;
    std::vector<gw_string_view_t> kmers;

    if (get_size(s) < kmer_size)
    {
        kmers.push_back(s);
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

float sequence_jaccard_similarity(const gw_string_view_t& a, const gw_string_view_t& b, const std::int32_t kmer_size, const std::int32_t stride)
{
    std::vector<gw_string_view_t> a_kmers = split_into_kmers(a, kmer_size, stride);
    std::vector<gw_string_view_t> b_kmers = split_into_kmers(b, kmer_size, stride);
    std::sort(std::begin(a_kmers), std::end(a_kmers));
    std::sort(std::begin(b_kmers), std::end(b_kmers));

    const std::size_t shared_kmers = count_shared_elements(a_kmers, b_kmers);
    // Calculate the set union size of a and b
    std::size_t union_size = a_kmers.size() + b_kmers.size() - shared_kmers;
    return static_cast<float>(shared_kmers) / static_cast<float>(union_size);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
