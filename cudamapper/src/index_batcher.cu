/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "index_batcher.cuh"

#include <algorithm>

namespace claragenomics
{
namespace cudamapper
{

std::vector<BatchOfIndices> generate_batches_of_indices(const number_of_indices_t query_indices_per_host_batch,
                                                        const number_of_indices_t query_indices_per_device_batch,
                                                        const number_of_indices_t target_indices_per_host_batch,
                                                        const number_of_indices_t target_indices_per_device_batch,
                                                        const std::shared_ptr<const claragenomics::io::FastaParser> query_parser,
                                                        const std::shared_ptr<const claragenomics::io::FastaParser> target_parser,
                                                        const number_of_basepairs_t query_basepairs_per_index,
                                                        const number_of_basepairs_t target_basepairs_per_index,
                                                        const bool same_query_and_target)
{
    if (same_query_and_target)
    {
        if (query_indices_per_host_batch != target_indices_per_host_batch)
        {
            throw std::invalid_argument("generate_batches_of_indices: indices_per_host_batch not the same");
        }
        if (query_indices_per_device_batch != target_indices_per_device_batch)
        {
            throw std::invalid_argument("generate_batches_of_indices: indices_per_device_batch not the same");
        }
        if (query_parser != target_parser)
        {
            throw std::invalid_argument("generate_batches_of_indices: parser not the same");
        }
        if (query_basepairs_per_index != target_basepairs_per_index)
        {
            throw std::invalid_argument("generate_batches_of_indices: basepairs_per_index not the same");
        }
    }

    // split indices into IndexDescriptors
    std::vector<IndexDescriptor> query_index_descriptors  = group_reads_into_indices(*query_parser,
                                                                                    query_basepairs_per_index);
    std::vector<IndexDescriptor> target_index_descriptors = group_reads_into_indices(*target_parser,
                                                                                     target_basepairs_per_index);

    // find host batches
    std::vector<IndexBatch> host_batches = details::index_batcher::group_into_batches(query_index_descriptors,
                                                                                      target_index_descriptors,
                                                                                      query_indices_per_host_batch,
                                                                                      target_indices_per_host_batch,
                                                                                      same_query_and_target);

    // create device batches for every host batch
    std::vector<BatchOfIndices> all_batches;
    for (IndexBatch& batch : host_batches)
    {
        // Device batches are only symmetric is their query and target indices are the same
        const bool same_query_and_target_in_batch = same_query_and_target && std::equal(begin(batch.query_indices),
                                                                                        end(batch.query_indices),
                                                                                        begin(batch.target_indices),
                                                                                        end(batch.target_indices));

        std::vector<IndexBatch> device_batches = details::index_batcher::group_into_batches(batch.query_indices,
                                                                                            batch.target_indices,
                                                                                            query_indices_per_device_batch,
                                                                                            target_indices_per_device_batch,
                                                                                            same_query_and_target_in_batch);

        all_batches.push_back({std::move(batch),
                               std::move(device_batches)});
    }

    return all_batches;
}

namespace details
{
namespace index_batcher
{

std::vector<IndexBatch> group_into_batches(const std::vector<IndexDescriptor>& query_indices,
                                           const std::vector<IndexDescriptor>& target_indices,
                                           const number_of_indices_t query_indices_per_batch,
                                           const number_of_indices_t target_indices_per_batch,
                                           const bool same_query_and_target)
{
    if (same_query_and_target)
    {
        if (query_indices_per_batch != target_indices_per_batch)
        {
            throw std::invalid_argument("split_batch: same_query_and_target is true, but indices_per_batch not the same.");
        }
    }

    std::vector<IndexBatch> batches;
    for (auto query_it = begin(query_indices); query_it < end(query_indices); query_it += query_indices_per_batch)
    {
        // if same_query_and_target only generate upper triangle of the query*target matrix instead of doing all-to-all
        auto t_begin = begin(target_indices) + (same_query_and_target ? distance(begin(query_indices), query_it) : 0);
        for (auto target_it = t_begin; target_it < end(target_indices); target_it += target_indices_per_batch)
        {
            std::vector<IndexDescriptor> batch_query_indices(query_it,
                                                             std::min(query_it + query_indices_per_batch,
                                                                      end(query_indices)));
            std::vector<IndexDescriptor> batch_target_indices(target_it,
                                                              std::min(target_it + target_indices_per_batch,
                                                                       end(target_indices)));

            batches.push_back({std::move(batch_query_indices),
                               std::move(batch_target_indices)});
        }
    }

    return batches;
}

} // namespace index_batcher
} // namespace details

} // namespace cudamapper
} // namespace claragenomics
