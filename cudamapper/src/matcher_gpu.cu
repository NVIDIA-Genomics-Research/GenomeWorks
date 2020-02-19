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

#include <claragenomics/utils/cudasort.cuh>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/mathutils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

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

namespace
{

template <typename RandomAccessIterator, typename ValueType>
__device__ RandomAccessIterator lower_bound(RandomAccessIterator lower_bound, RandomAccessIterator upper_bound, ValueType query)
{
    assert(upper_bound >= lower_bound);
    while (upper_bound > lower_bound)
    {
        RandomAccessIterator mid = lower_bound + (upper_bound - lower_bound) / 2;
        const auto mid_value     = *mid;
        if (mid_value < query)
            lower_bound = mid + 1;
        else
            upper_bound = mid;
    }
    return lower_bound;
}

template <typename RandomAccessIterator, typename ValueType>
__device__ RandomAccessIterator upper_bound(RandomAccessIterator lower_bound, RandomAccessIterator upper_bound, ValueType query)
{
    assert(upper_bound >= lower_bound);
    while (upper_bound > lower_bound)
    {
        RandomAccessIterator mid = lower_bound + (upper_bound - lower_bound) / 2;
        const auto mid_value     = *mid;
        if (mid_value <= query)
            lower_bound = mid + 1;
        else
            upper_bound = mid;
    }
    return lower_bound;
}

/// \brief Generates an array of anchors from matches of representations of the query and target index
///
/// See generate_anchors_dispatcher() for more details
///
/// \param anchors_d the array to be filled with anchors, the size of this array has to be equal to the last element of anchor_starting_indices
/// \param compound_key_read_ids_d the array to be filled with compund keys, the size of this array has to be equal to the last element of anchor_starting_indices
/// \param compound_key_positions_in_reads_d the array to be filled with compund keys, the size of this array has to be equal to the last element of anchor_starting_indices
/// \param n_anchors the size of the anchors array
/// \param anchor_starting_indices_d the array of starting indices of the set of anchors for each unique representation of the query index (representations with no match in target will have the same starting index as the last matching representation)
/// \param query_starting_index_of_each_representation_d the starting index of a representation in query_read_ids and query_positions_in_read
/// \param found_target_indices_d the found matches in the array of unique target representation for each unique representation of query index
/// \param target_starting_index_of_each_representation_d the starting index of a representation in target_read_ids and target_positions_in_read
/// \param n_query_representations the size of the query_starting_index_of_each_representation_d and found_target_indices_d arrays, ie. the number of unique representations in the query index
/// \param query_read_ids the array of read ids of the (read id, position)-pairs in query index
/// \param query_positions_in_read the array of positions of the (read id, position)-pairs in query index
/// \param target_read_ids the array of read ids of the (read id, position)-pairs in target index
/// \param target_positions_in_read the array of positions of the (read id, position)-pairs in target index
/// \param smallest_query_read_id smallest read_id in query index
/// \param smallest_target_read_id smallest read_id in target index
/// \param number_of_target_reads number of read_ids in taget index
/// \param max_basepairs_in_target_reads number of basepairs in longest read in target index
/// \tparam ReadsKeyT type of compound_key_read_ids_d, has to be integral
/// \tparam PositionsKeyT type of compound_key_positions_in_reads_d, has to be integral
template <typename ReadsKeyT, typename PositionsKeyT>
__global__ void generate_anchors_kernel(
    Anchor* const anchors_d,
    ReadsKeyT* const compound_key_read_ids_d,
    PositionsKeyT* const compound_key_positions_in_reads_d,
    const std::int64_t n_anchors,
    const std::int64_t* const anchor_starting_index_d,
    const std::uint32_t* const query_starting_index_of_each_representation_d,
    const std::int64_t* const found_target_indices_d,
    const std::int32_t n_query_representations,
    const std::uint32_t* const target_starting_index_of_each_representation_d,
    const read_id_t* const query_read_ids,
    const position_in_read_t* const query_positions_in_read,
    const read_id_t* const target_read_ids,
    const position_in_read_t* const target_positions_in_read,
    const read_id_t smallest_query_read_id,
    const read_id_t smallest_target_read_id,
    const read_id_t number_of_target_reads,
    const position_in_read_t max_basepairs_in_target_reads)
{
    // Fill the anchor_d array. Each thread generates one anchor.
    std::int64_t anchor_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (anchor_idx >= n_anchors)
        return;

    // Figure out for which representation this thread should compute the anchor.
    // We only need the index in the unique representation array of the query index
    // not the representation itself.
    const std::int64_t representation_idx = upper_bound(anchor_starting_index_d, anchor_starting_index_d + n_query_representations, anchor_idx) - anchor_starting_index_d;

    assert(representation_idx < n_query_representations);

    // Compute the index of the anchor within only this representation.
    std::uint32_t relative_anchor_index = anchor_idx;
    if (representation_idx > 0)
        relative_anchor_index -= anchor_starting_index_d[representation_idx - 1];

    // Get the ranges within the query and target index with this representation.
    const std::int64_t j = found_target_indices_d[representation_idx];
    assert(j >= 0);
    const std::uint32_t query_begin  = query_starting_index_of_each_representation_d[representation_idx];
    const std::uint32_t target_begin = target_starting_index_of_each_representation_d[j];
    const std::uint32_t target_end   = target_starting_index_of_each_representation_d[j + 1];

    const std::uint32_t n_targets = target_end - target_begin;

    // Overall we want to do an all-to-all (n*m) matching between the query and target entries
    // with the same representation.
    // Compute the exact combination query and target index entry for which
    // we generate the anchor in this thread.
    const std::uint32_t query_idx  = query_begin + relative_anchor_index / n_targets;
    const std::uint32_t target_idx = target_begin + relative_anchor_index % n_targets;

    assert(query_idx < query_starting_index_of_each_representation_d[representation_idx + 1]);

    // Generate and store the anchor
    claragenomics::cudamapper::Anchor a;
    a.query_read_id_           = query_read_ids[query_idx];
    a.target_read_id_          = target_read_ids[target_idx];
    a.query_position_in_read_  = query_positions_in_read[query_idx];
    a.target_position_in_read_ = target_positions_in_read[target_idx];
    anchors_d[anchor_idx]      = a;

    // Calculate compound keys
    // Encode keys as query_x * max_target_x + target_x.
    // query_x * max_target_x is similar to shifting query_x by the number of bits needed to store largest possible target_x
    // As for most indices read_id will not start from 0 use "local" read_ids to make the keys shorter
    // (local_read_id = real_read_id - smallest_read_id)
    // Reason for cast: if both multiplication inputs are 32-bit the result will also be 32-bit, even if it is to be stored
    // in a 64-bit variable. This could cause an overflow. Casting one of them to 64-bit makes the result also be 64-bit.
    // It's up to the user to decide if the output should be 32 or 64-bit.
    compound_key_read_ids_d[anchor_idx]           = (a.query_read_id_ - smallest_query_read_id) * static_cast<ReadsKeyT>(number_of_target_reads) + (a.target_read_id_ - smallest_target_read_id);
    compound_key_positions_in_reads_d[anchor_idx] = a.query_position_in_read_ * static_cast<PositionsKeyT>(max_basepairs_in_target_reads) + a.target_position_in_read_;
}

/// \brief Generates anchors
///
/// See generate_anchors_dispatcher() for details
///
/// \param anchors the array to be filled with anchors, the size of this array has to be equal to the last element of anchor_starting_indices
/// \param anchor_starting_indices_d the array of starting indices of the set of anchors for each unique representation of the query index (representations with no match in target will have the same starting index as the last matching representation)
/// \param found_target_indices_d the found matches in the array of unique target representation for each unique representation of query index
/// \param query_index
/// \param target_index
/// \param max_reads_compound_key largest possible read_id compound key
/// \param max_positions_compound_key largest possible position_in_read compund key
/// \tparam ReadsKeyT type of compound_key_read_ids, has to be integral
/// \tparam PositionsKeyT type of compound_key_positions_in_reads, has to be integral
template <typename ReadsKeyT, typename PositionsKeyT>
void generate_anchors(
    device_buffer<Anchor>& anchors,
    const device_buffer<std::int64_t>& anchor_starting_indices_d,
    const device_buffer<std::int64_t>& found_target_indices_d,
    const Index& query_index,
    const Index& target_index,
    const std::uint64_t max_reads_compound_key,
    const std::uint64_t max_positions_compound_key)
{
    static_assert(std::is_integral<ReadsKeyT>::value, "ReadsKeyT has to be integral");
    static_assert(std::is_integral<PositionsKeyT>::value, "PositionsKeyT has to be integral");

    const device_buffer<std::uint32_t>& query_starting_index_of_each_representation_d = query_index.first_occurrence_of_representations();
    const device_buffer<read_id_t>& query_read_ids                                    = query_index.read_ids();
    const device_buffer<position_in_read_t>& query_positions_in_read                  = query_index.positions_in_reads();

    const device_buffer<std::uint32_t>& target_starting_index_of_each_representation_d = target_index.first_occurrence_of_representations();
    const device_buffer<cudamapper::read_id_t>& target_read_ids                        = target_index.read_ids();
    const device_buffer<cudamapper::position_in_read_t>& target_positions_in_read      = target_index.positions_in_reads();

    assert(anchor_starting_indices_d.size() + 1 == query_starting_index_of_each_representation_d.size());
    assert(found_target_indices_d.size() + 1 == query_starting_index_of_each_representation_d.size());
    assert(query_read_ids.size() == query_positions_in_read.size());
    assert(target_read_ids.size() == target_positions_in_read.size());

    // TODO: Using CudaMallocAllocator for now. Switch to using the allocator used by input arrays
    //       once device_buffer::get_allocator() is added
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CudaMallocAllocator>();

    device_buffer<ReadsKeyT> compound_key_read_ids(anchors.size(), allocator);
    device_buffer<PositionsKeyT> compound_key_positions_in_reads(anchors.size(), allocator);

    {
        CGA_NVTX_RANGE(profile, "matcherGPU::generate_anchors_kernel");
        const int32_t n_threads = 256;
        const int32_t n_blocks  = claragenomics::ceiling_divide<int64_t>(get_size(anchors), n_threads);
        generate_anchors_kernel<<<n_blocks, n_threads>>>(
            anchors.data(),
            compound_key_read_ids.data(),
            compound_key_positions_in_reads.data(),
            get_size(anchors),
            anchor_starting_indices_d.data(),
            query_starting_index_of_each_representation_d.data(),
            found_target_indices_d.data(),
            get_size(found_target_indices_d),
            target_starting_index_of_each_representation_d.data(),
            query_read_ids.data(),
            query_positions_in_read.data(),
            target_read_ids.data(),
            target_positions_in_read.data(),
            query_index.smallest_read_id(),
            target_index.smallest_read_id(),
            target_index.number_of_reads(),
            target_index.number_of_basepairs_in_longest_read());
    }

    {
        CGA_NVTX_RANGE(profile, "matcherGPU::sort_anchors");
        // sort anchors by query_read_id -> target_read_id -> query_position_in_read -> target_position_in_read
        cudautils::sort_by_two_keys(compound_key_read_ids,
                                    compound_key_positions_in_reads,
                                    anchors,
                                    static_cast<ReadsKeyT>(max_reads_compound_key),
                                    static_cast<PositionsKeyT>(max_positions_compound_key));
    }
}

} // namespace

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

            generate_anchors<ReadsKeyT, PositionsKeyT>(anchors,
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

            generate_anchors<ReadsKeyT, PositionsKeyT>(anchors,
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

            generate_anchors<ReadsKeyT, PositionsKeyT>(anchors,
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

            generate_anchors<ReadsKeyT, PositionsKeyT>(anchors,
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
