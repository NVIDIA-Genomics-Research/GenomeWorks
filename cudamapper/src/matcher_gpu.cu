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
    return anchors_h_;
}

namespace details
{

namespace matcher_gpu
{
thrust::device_vector<std::uint32_t> find_first_occurrences_of_representations(const thrust::device_vector<representation_t>& representations_d)
{
    // each element has value 1 if representation with the same index in representations_d has a different value than it's neighbour to the left, 0 otehrwise
    // underlying type is 32-bit because a scan operation will be performed on the array, so the elements should be capable of holding a number that is equal to
    // the total number of 1s in the array
    thrust::device_vector<std::uint32_t> new_value_mask_d(representations_d.size());

    // TODO: Currently maximum number of thread blocks is 2^31-1. This means we support representations of up to (2^31-1) * number_of_threads
    // With 256 that's (2^31-1)*2^8 ~= 2^39. If representation is 4-byte (we expect it to be 4 or 8) that's 2^39*2^2 = 2^41 = 2TB. We don't expect to hit this limit any time soon
    // The kernel can be modified to process several representation per thread to support arbitrary size
    std::uint32_t number_of_threads = 256; // arbitrary value
    std::uint32_t number_of_blocks  = (representations_d.size() - 1) / number_of_threads + 1;

    create_new_value_mask<<<number_of_blocks, number_of_threads>>>(thrust::raw_pointer_cast(representations_d.data()),
                                                                   representations_d.size(),
                                                                   thrust::raw_pointer_cast(new_value_mask_d.data()));
    CGA_CU_CHECK_ERR(cudaDeviceSynchronize()); // sync not necessary, here only to detect the error immediately

    // do inclusive scan
    // for example for
    // 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
    // 0  0  0  0 12 12 12 12 12 12 23 23 23 32 32 32 32 32 46 46 46
    // 1  0  0  0  1  0  0  0  0  0  1  0  0  1  0  0  0  0  1  0  0
    // gives
    // 1  1  1  1  2  2  2  2  2  2  3  3  3  4  4  4  4  4  5  5  5
    // meaning all elements with the same representation have the same value and those values are sorted in increasing order starting from 1
    thrust::device_vector<std::uint64_t> representation_index_mask_d(new_value_mask_d.size());
    thrust::inclusive_scan(thrust::device,
                           new_value_mask_d.begin(),
                           new_value_mask_d.end(),
                           representation_index_mask_d.begin());
    new_value_mask_d.clear();
    new_value_mask_d.shrink_to_fit();

    std::uint64_t number_of_unique_representations = representation_index_mask_d.back(); // D2H copy

    thrust::device_vector<std::uint32_t> starting_index_of_each_representation(number_of_unique_representations + 1);

    copy_index_of_first_occurence<<<number_of_blocks, number_of_threads>>>(thrust::raw_pointer_cast(representation_index_mask_d.data()),
                                                                           representation_index_mask_d.size(),
                                                                           thrust::raw_pointer_cast(starting_index_of_each_representation.data()));
    // last element is the total number of elements in representations array
    starting_index_of_each_representation.back() = representations_d.size(); // H2D copy

    return starting_index_of_each_representation;
}

void find_query_target_matches(thrust::device_vector<std::int64_t>& found_target_indices_d, const thrust::device_vector<representation_t>& query_representations_d, const thrust::device_vector<representation_t>& target_representations_d)
{
    assert(found_target_indices_d.size() == query_representations_d.size());

    const int32_t n_threads = 256;
    const int32_t n_blocks  = ceiling_divide<int64_t>(query_representations_d.size(), n_threads);

    find_query_target_matches_kernel<<<n_blocks, n_threads>>>(found_target_indices_d.data().get(), query_representations_d.data().get(), get_size(query_representations_d), target_representations_d.data().get(), get_size(target_representations_d));
}

void compute_anchor_starting_indices(thrust::device_vector<std::int64_t>& anchor_starting_indices_d, const thrust::device_vector<std::uint32_t> query_starting_index_of_each_representation_d, const thrust::device_vector<std::int64_t>& found_target_indices_d, const thrust::device_vector<std::uint32_t> target_starting_index_of_each_representation_d)
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

__global__ void create_new_value_mask(const representation_t* const representations_d,
                                      const std::size_t number_of_elements,
                                      std::uint32_t* const new_value_mask_d)
{
    std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= number_of_elements)
        return;

    if (index == 0)
    {
        new_value_mask_d[0] = 1;
    }
    else
    {
        if (representations_d[index] == representations_d[index - 1])
        {
            new_value_mask_d[index] = 0;
        }
        else
            new_value_mask_d[index] = 1;
    }
}

__global__ void copy_index_of_first_occurence(const std::uint64_t* const representation_index_mask_d,
                                              const std::size_t number_of_input_elements,
                                              std::uint32_t* const starting_index_of_each_representation)
{
    std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= number_of_input_elements)
        return;

    if (index == 0)
    {
        starting_index_of_each_representation[0] = 0;
    }
    else
    {
        if (representation_index_mask_d[index] != representation_index_mask_d[index - 1])
        {
            // if new representation (= not the same as its left neighbor)
            // save the index at which that representation starts
            // representation_index_mask_d gives a unique index to each representation, starting from 1, thus '-1'
            starting_index_of_each_representation[representation_index_mask_d[index] - 1] = index;
        }
    }
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

} // namespace matcher_gpu

} // namespace details
} // namespace cudamapper

} // namespace claragenomics
