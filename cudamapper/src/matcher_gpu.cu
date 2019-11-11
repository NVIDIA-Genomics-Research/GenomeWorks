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

#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/mathutils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claragenomics
{

namespace cudamapper
{

MatcherGPU::MatcherGPU(const IndexTwoIndices& query_index,
                       const IndexTwoIndices& target_index)
{
}

thrust::device_vector<Anchor>& MatcherGPU::anchors()
{
    return anchors_d_;
}

namespace details
{

namespace matcher_gpu
{

void find_query_target_matches(thrust::device_vector<std::int64_t>& found_target_indices_d, const thrust::device_vector<representation_t>& query_representations_d, const thrust::device_vector<representation_t>& target_representations_d)
{
    assert(found_target_indices_d.size() == query_representations_d.size());

    const int32_t n_threads = 256;
    const int32_t n_blocks  = ceiling_divide<int64_t>(query_representations_d.size(), n_threads);

    find_query_target_matches_kernel<<<n_blocks, n_threads>>>(found_target_indices_d.data().get(), query_representations_d.data().get(), get_size(query_representations_d), target_representations_d.data().get(), get_size(target_representations_d));
}

void compute_anchor_starting_indices(thrust::device_vector<std::int64_t>& anchor_starting_indices_d,
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

__global__ void find_query_target_matches_kernel(int64_t* const found_target_indices, const representation_t* const query_representations_d, const int64_t n_query_representations, const representation_t* const target_representations_d, const int64_t n_target_representations)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_query_representations)
        return;

    const representation_t query        = query_representations_d[i];
    const representation_t* lower_bound = target_representations_d;
    const representation_t* upper_bound = target_representations_d + n_target_representations;
    int64_t found_target_index          = -1;
    while (upper_bound - lower_bound > 0)
    {
        const representation_t* mid   = lower_bound + (upper_bound - lower_bound) / 2;
        const representation_t target = *mid;
        if (target < query)
            lower_bound = mid + 1;
        else if (target > query)
            upper_bound = mid;
        else
        {
            found_target_index = mid - target_representations_d;
            break;
        }
    }

    found_target_indices[i] = found_target_index;
}

void generate_anchors(thrust::device_vector<Anchor>& anchors,
                      const thrust::device_vector<std::int64_t>& anchor_starting_indices,
                      const thrust::device_vector<std::uint32_t>& query_starting_index_of_each_representation_d,
                      const thrust::device_vector<std::int64_t>& found_target_indices_d,
                      const thrust::device_vector<std::uint32_t>& target_starting_index_of_each_representation_d,
                      const thrust::device_vector<read_id_t>& query_read_ids,
                      const thrust::device_vector<position_in_read_t>& query_positions_in_read,
                      const thrust::device_vector<read_id_t>& target_read_ids,
                      const thrust::device_vector<position_in_read_t>& target_positions_in_read)
{
    assert(anchor_starting_indices.size() + 1 == query_starting_index_of_each_representation_d.size());
    assert(found_target_indices_d.size() + 1 == query_starting_index_of_each_representation_d.size());
    assert(query_read_ids.size() == query_positions_in_read.size());
    assert(target_read_ids.size() == target_positions_in_read.size());

    const int32_t n_threads = 256;
    const int32_t n_blocks  = ceiling_divide<int64_t>(get_size(found_target_indices_d), n_threads);
    generate_anchors_kernel<<<n_blocks, n_threads>>>(
        anchors.data().get(),
        anchor_starting_indices.data().get(),
        query_starting_index_of_each_representation_d.data().get(),
        found_target_indices_d.data().get(),
        get_size(found_target_indices_d),
        target_starting_index_of_each_representation_d.data().get(),
        query_read_ids.data().get(),
        query_positions_in_read.data().get(),
        target_read_ids.data().get(),
        target_positions_in_read.data().get());
}

__global__ void generate_anchors_kernel(
    Anchor* const anchors_d,
    const int64_t* const anchor_starting_index_d,
    const std::uint32_t* const query_starting_index_of_each_representation_d,
    const std::int64_t* const found_target_indices_d,
    int32_t n_query_representations,
    const std::uint32_t* const target_starting_index_of_each_representation_d,
    const read_id_t* const query_read_ids,
    const position_in_read_t* const query_positions_in_read,
    const read_id_t* const target_read_ids,
    const position_in_read_t* const target_positions_in_read)
{
    const std::int32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n_query_representations)
        return;

    const std::int64_t j = found_target_indices_d[i];
    if (j < 0)
        return;

    std::int64_t anchor_idx = 0;
    if (i > 0)
        anchor_idx = anchor_starting_index_d[i - 1];
    std::uint32_t query_idx          = query_starting_index_of_each_representation_d[i];
    const std::uint32_t query_end    = query_starting_index_of_each_representation_d[i + 1];
    const std::uint32_t target_begin = target_starting_index_of_each_representation_d[j];
    const std::uint32_t target_end   = target_starting_index_of_each_representation_d[j + 1];

    while (query_idx < query_end)
    {
        std::uint32_t target_idx = target_begin;
        while (target_idx < target_end)
        {
            Anchor a;
            a.query_read_id_           = query_read_ids[query_idx];
            a.target_read_id_          = target_read_ids[target_idx];
            a.query_position_in_read_  = query_positions_in_read[query_idx];
            a.target_position_in_read_ = target_positions_in_read[target_idx];
            anchors_d[anchor_idx]      = a;
            ++anchor_idx;
            ++target_idx;
        }
        ++query_idx;
    }
    assert(anchor_idx == anchor_starting_index_d[i] || anchor_starting_index_d[i - 1] == anchor_starting_index_d[i]);
}

} // namespace matcher_gpu

} // namespace details
} // namespace cudamapper

} // namespace claragenomics
