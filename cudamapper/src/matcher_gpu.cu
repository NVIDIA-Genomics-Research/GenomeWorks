/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "matcher_gpu.cuh"

#include <cassert>
#include <numeric>

#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>

namespace claragenomics
{

namespace cudamapper
{

MatcherGPU::MatcherGPU(std::shared_ptr<DeviceAllocator> allocator,
                       const Index& query_index,
                       const Index& target_index)
    : anchors_d_(allocator)
{
    CGA_NVTX_RANGE(profile, "matcherGPU");
    if (query_index.unique_representations().size() == 0 || target_index.unique_representations().size() == 0)
        return;

    // We need to compute a set of anchors between the query and the target.
    // An anchor is a combination of a query (read_id, position) and
    // target {read_id, position} with the same representation.
    // The set of anchors of a matching query and target representation
    // is the all-to-all combination of the corresponding set of {(read_id, position)}
    // of the query with the set of {(read_id, position)} of the target.
    //
    // We compute the anchors for each unique representation of the query index.
    // The array index of the following data structures will correspond to the array index of the
    // unique representation in the query index.

    device_buffer<std::int64_t> found_target_indices_d(query_index.unique_representations().size(), allocator);
    device_buffer<std::int64_t> anchor_starting_indices_d(query_index.unique_representations().size(), allocator);

    // First we search for each unique representation of the query index, the array index
    // of the same representation in the array of unique representations of target index
    // (or -1 if representation is not found).
    details::matcher_gpu::find_query_target_matches(found_target_indices_d, query_index.unique_representations(), target_index.unique_representations());

    // For each unique representation of the query index compute the number of corrsponding anchors
    // and store the resulting starting index in an anchors array if all anchors are stored in a flat array.
    // The last element will be the total number of anchors.
    details::matcher_gpu::compute_anchor_starting_indices(anchor_starting_indices_d, query_index.first_occurrence_of_representations(), found_target_indices_d, target_index.first_occurrence_of_representations());

    const int64_t n_anchors = cudautils::get_value_from_device(anchor_starting_indices_d.end() - 1); // D->H transfer

    anchors_d_.resize(n_anchors);

    // Generate the anchors
    // by computing the all-to-all combinations of the matching representations in query and target
    details::matcher_gpu::generate_anchors_dispatcher(anchors_d_,
                                                      anchor_starting_indices_d,
                                                      found_target_indices_d,
                                                      query_index,
                                                      target_index);
}

device_buffer<Anchor>& MatcherGPU::anchors()
{
    return anchors_d_;
}

namespace details
{

namespace matcher_gpu
{

void find_query_target_matches(
    device_buffer<std::int64_t>& found_target_indices_d,
    const device_buffer<representation_t>& query_representations_d,
    const device_buffer<representation_t>& target_representations_d)
{
    assert(found_target_indices_d.size() == query_representations_d.size());

    const int32_t n_threads = 256;
    const int32_t n_blocks  = ceiling_divide<int64_t>(query_representations_d.size(), n_threads);

    find_query_target_matches_kernel<<<n_blocks, n_threads>>>(found_target_indices_d.data(), query_representations_d.data(), get_size(query_representations_d), target_representations_d.data(), get_size(target_representations_d));
}

void compute_anchor_starting_indices(
    device_buffer<std::int64_t>& anchor_starting_indices_d,
    const device_buffer<std::uint32_t>& query_starting_index_of_each_representation_d,
    const device_buffer<std::int64_t>& found_target_indices_d,
    const device_buffer<std::uint32_t>& target_starting_index_of_each_representation_d)
{
    assert(query_starting_index_of_each_representation_d.size() == found_target_indices_d.size() + 1);
    assert(anchor_starting_indices_d.size() == found_target_indices_d.size());

    const std::uint32_t* const query_starting_indices  = query_starting_index_of_each_representation_d.data();
    const std::uint32_t* const target_starting_indices = target_starting_index_of_each_representation_d.data();
    const std::int64_t* const found_target_indices     = found_target_indices_d.data();

    thrust::transform_inclusive_scan(
        thrust::device,
        thrust::make_counting_iterator(std::int64_t(0)),
        thrust::make_counting_iterator(get_size(anchor_starting_indices_d)),
        anchor_starting_indices_d.begin(),
        [query_starting_indices, target_starting_indices, found_target_indices] __device__(std::uint32_t query_index) -> std::int64_t {
            std::int32_t n_queries_with_representation = query_starting_indices[query_index + 1] - query_starting_indices[query_index];
            std::int64_t target_index                  = found_target_indices[query_index];
            std::int32_t n_targets_with_representation = 0;
            if (target_index >= 0)
                n_targets_with_representation = target_starting_indices[target_index + 1] - target_starting_indices[target_index];
            return n_queries_with_representation * n_targets_with_representation;
        },
        thrust::plus<std::int64_t>());
}

void generate_anchors_dispatcher(
    device_buffer<Anchor>& anchors,
    const device_buffer<std::int64_t>& anchor_starting_indices_d,
    const device_buffer<std::int64_t>& found_target_indices_d,
    const Index& query_index,
    const Index& target_index)
{
    const read_id_t number_of_query_reads                  = query_index.number_of_reads();
    const read_id_t number_of_target_reads                 = target_index.number_of_reads();
    const position_in_read_t max_basepairs_in_query_reads  = query_index.number_of_basepairs_in_longest_read();
    const position_in_read_t max_basepairs_in_target_reads = target_index.number_of_basepairs_in_longest_read();

    std::uint64_t max_reads_compound_key     = number_of_query_reads * static_cast<std::uint64_t>(number_of_target_reads) + number_of_target_reads;
    std::uint64_t max_positions_compound_key = max_basepairs_in_query_reads * static_cast<std::uint64_t>(max_basepairs_in_target_reads) + max_basepairs_in_target_reads;

    // TODO: This solution with four separate calls depending on max key sizes ir rather messy.
    //       Look for a solution similar to std::conditional, but which can be done at runtime.

    bool reads_compound_key_32_bit     = max_reads_compound_key <= std::numeric_limits<std::uint32_t>::max();
    bool positions_compound_key_32_bit = max_positions_compound_key <= std::numeric_limits<std::uint32_t>::max();

    if (reads_compound_key_32_bit)
    {
        using ReadsKeyT = std::uint32_t;
        if (positions_compound_key_32_bit)
        {
            using PositionsKeyT = std::uint32_t;

            details::matcher_gpu::generate_anchors<ReadsKeyT, PositionsKeyT>(anchors,
                                                                             anchor_starting_indices_d,
                                                                             found_target_indices_d,
                                                                             query_index,
                                                                             target_index,
                                                                             max_reads_compound_key,
                                                                             max_positions_compound_key);
        }
        else
        {
            using PositionsKeyT = std::uint64_t;

            details::matcher_gpu::generate_anchors<ReadsKeyT, PositionsKeyT>(anchors,
                                                                             anchor_starting_indices_d,
                                                                             found_target_indices_d,
                                                                             query_index,
                                                                             target_index,
                                                                             max_reads_compound_key,
                                                                             max_positions_compound_key);
        }
    }
    else
    {
        using ReadsKeyT = std::uint64_t;
        if (positions_compound_key_32_bit)
        {
            using PositionsKeyT = std::uint32_t;

            details::matcher_gpu::generate_anchors<ReadsKeyT, PositionsKeyT>(anchors,
                                                                             anchor_starting_indices_d,
                                                                             found_target_indices_d,
                                                                             query_index,
                                                                             target_index,
                                                                             max_reads_compound_key,
                                                                             max_positions_compound_key);
        }
        else
        {
            using PositionsKeyT = std::uint64_t;

            details::matcher_gpu::generate_anchors<ReadsKeyT, PositionsKeyT>(anchors,
                                                                             anchor_starting_indices_d,
                                                                             found_target_indices_d,
                                                                             query_index,
                                                                             target_index,
                                                                             max_reads_compound_key,
                                                                             max_positions_compound_key);
        }
    }
}

__global__ void find_query_target_matches_kernel(
    int64_t* const found_target_indices,
    const representation_t* const query_representations_d,
    const int64_t n_query_representations,
    const representation_t* const target_representations_d,
    const int64_t n_target_representations)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_query_representations)
        return;

    const representation_t query = query_representations_d[i];
    int64_t found_target_index   = -1;
    const representation_t* lb   = lower_bound(target_representations_d, target_representations_d + n_target_representations, query);
    if (*lb == query)
        found_target_index = lb - target_representations_d;

    found_target_indices[i] = found_target_index;
}

} // namespace matcher_gpu

} // namespace details
} // namespace cudamapper

} // namespace claragenomics
