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

#include <thrust/scan.h>
#include <thrust/transform_scan.h>
#include <thrust/execution_policy.h>
#include <cassert>

#include <claragenomics/utils/cudautils.hpp>

namespace claragenomics
{

namespace cudamapper
{

MatcherGPU::MatcherGPU(const Index& query_index,
                       const Index& target_index)
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

    thrust::device_vector<std::int64_t> found_target_indices_d(query_index.unique_representations().size());
    thrust::device_vector<std::int64_t> anchor_starting_indices_d(query_index.unique_representations().size());

    // First we search for each unique representation of the query index, the array index
    // of the same representation in the array of unique representations of target index
    // (or -1 if representation is not found).
    details::matcher_gpu::find_query_target_matches(found_target_indices_d, query_index.unique_representations(), target_index.unique_representations());

    // For each unique representation of the query index compute the number of corrsponding anchors
    // and store the resulting starting index in an anchors array if all anchors are stored in a flat array.
    // The last element will be the total number of anchors.
    details::matcher_gpu::compute_anchor_starting_indices(anchor_starting_indices_d, query_index.first_occurrence_of_representations(), found_target_indices_d, target_index.first_occurrence_of_representations());

    const int64_t n_anchors = anchor_starting_indices_d.back(); // D->H transfer

    anchors_d_.resize(n_anchors);

    thrust::device_vector<std::uint64_t> compound_key_read_ids_d;
    thrust::device_vector<std::uint64_t> compound_key_positions_in_reads_d;

    // Generate the anchors
    // by computing the all-to-all combinations of the matching representations in query and target
    details::matcher_gpu::generate_anchors(anchors_d_,
                                           compound_key_read_ids_d,
                                           compound_key_positions_in_reads_d,
                                           anchor_starting_indices_d,
                                           query_index.first_occurrence_of_representations(),
                                           found_target_indices_d,
                                           target_index.first_occurrence_of_representations(),
                                           query_index.read_ids(),
                                           query_index.positions_in_reads(),
                                           target_index.read_ids(),
                                           target_index.positions_in_reads(),
                                           query_index.smallest_read_id(),
                                           target_index.smallest_read_id(),
                                           target_index.number_of_reads(),
                                           target_index.number_of_basepairs_in_longest_read());
}

thrust::device_vector<Anchor>& MatcherGPU::anchors()
{
    return anchors_d_;
}

namespace details
{

namespace matcher_gpu
{

void find_query_target_matches(
    thrust::device_vector<std::int64_t>& found_target_indices_d,
    const thrust::device_vector<representation_t>& query_representations_d,
    const thrust::device_vector<representation_t>& target_representations_d)
{
    assert(found_target_indices_d.size() == query_representations_d.size());

    const int32_t n_threads = 256;
    const int32_t n_blocks  = ceiling_divide<int64_t>(query_representations_d.size(), n_threads);

    find_query_target_matches_kernel<<<n_blocks, n_threads>>>(found_target_indices_d.data().get(), query_representations_d.data().get(), get_size(query_representations_d), target_representations_d.data().get(), get_size(target_representations_d));
}

void compute_anchor_starting_indices(
    thrust::device_vector<std::int64_t>& anchor_starting_indices_d,
    const thrust::device_vector<std::uint32_t>& query_starting_index_of_each_representation_d,
    const thrust::device_vector<std::int64_t>& found_target_indices_d,
    const thrust::device_vector<std::uint32_t>& target_starting_index_of_each_representation_d)
{
    assert(query_starting_index_of_each_representation_d.size() == found_target_indices_d.size() + 1);
    assert(anchor_starting_indices_d.size() == found_target_indices_d.size());

    const std::uint32_t* const query_starting_indices  = query_starting_index_of_each_representation_d.data().get();
    const std::uint32_t* const target_starting_indices = target_starting_index_of_each_representation_d.data().get();
    const std::int64_t* const found_target_indices     = found_target_indices_d.data().get();

    thrust::transform_inclusive_scan(
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
