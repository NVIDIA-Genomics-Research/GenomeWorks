/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <vector>

#include "index_cache.cuh"
#include "index_descriptor.hpp"

#include <claragenomics/io/fasta_parser.hpp>

namespace claragenomics
{
namespace cudamapper
{

/// IndexBatch
///
/// IndexBatch consists of sets of query and target indices that belong to one batch, either host or device
struct IndexBatch
{
    std::vector<IndexDescriptor> query_indices;
    std::vector<IndexDescriptor> target_indices;
};

/// BatchOfIndices
///
/// BatchOfIndices represents one batch of query and target indices that should be saved in host memory.
/// device_batches contains batches of indices that are parts of host_batch. These batches should be loaded into
/// device memory one by one from host memory
struct BatchOfIndices
{
    IndexBatch host_batch;
    std::vector<IndexBatch> device_batches;
};

/// number_of_indices_t
using number_of_indices_t = std::int32_t;

/// \brief Groups indices into batches
///
/// This function groups indices into batches. Host batch contains one section of query and one section of device indices.
/// Every device batch contains one query subsection and one target subsection of its host host batch
///
/// If same_query_and_target is false every section of query indices is combined with ever section of target indices.
/// If same_query_and_target is true sections of target indices are only combined with sections of target indices with smaller section id.
/// This is done because on that case due to symmetry having (query_5, target_7) is equivalent to (target_7, query_5)
///
/// For example imagine that both query and target sections of indices are ((0, 10), (10, 10)), ((20, 10), (30, 10))
/// If same_query_and_target == false generated host batches would be:
/// q(( 0, 10), (10, 10)), t(( 0, 10), (10, 10))
/// q(( 0, 10), (10, 10)), t((20, 10), (30, 10))
/// q((20, 10), (30, 10)), t(( 0, 10), (10, 10))
/// q((20, 10), (30, 10)), t((20, 10), (30, 10))
/// If same_query_and_target == true generated host batches would be:
/// q(( 0, 10), (10, 10)), t(( 0, 10), (10, 10))
/// q(( 0, 10), (10, 10)), t((20, 10), (30, 10))
/// q((20, 10), (30, 10)), t((20, 10), (30, 10))
/// i.e. q((20, 10), (30, 10)), t(( 0, 10), (10, 10)) would be missing beacuse it is already coveder by q(( 0, 10), (10, 10)), t((20, 10), (30, 10)) by symmetry
///
/// The same holds for device batches in every generated host batch in which query and target sections are the same
/// If same_query_and_target == true in the case above the follwoing device batches would be generated (assuming that every device batch has only one index)
/// For q(( 0, 10), (10, 10)), t(( 0, 10), (10, 10)):
/// q( 0, 10), t( 0, 10)
/// q( 0, 10), t(10, 10)
/// skipping q( 10, 10), t( 0, 10) due to symmetry with q( 0, 10), t(10, 10)
/// q(10, 10), t(10, 10)
/// For q(( 0, 10), (10, 10)), t((20, 10), (30, 10))
/// q( 0, 10), t(20, 10)
/// q( 0, 10), t(30, 10)
/// q(10, 10), t(20, 10)
/// q(10, 10), t(30, 10)
/// For q((20, 10), (30, 10)), t((20, 10), (30, 10))
/// q(20, 10), t(20, 10)
/// q(20, 10), t(30, 10)
/// skipping q(30, 10), t(20, 10) due to symmetry with q( 20, 10), t(30, 10)
/// q(30, 10), t(30, 10)
///
/// \param query_indices_per_host_batch
/// \param query_indices_per_device_batch
/// \param target_indices_per_host_batch
/// \param target_indices_per_device_batch
/// \param query_parser
/// \param target_parser
/// \param query_basepairs_per_index
/// \param target_basepairs_per_index
/// \param same_query_and_target
/// \throw std::invalid_argument if same_query_and_target is true and corresponding parameters for query and target are not the same
/// \return generated batches
std::vector<BatchOfIndices> generate_batches_of_indices(number_of_indices_t query_indices_per_host_batch,
                                                        number_of_indices_t query_indices_per_device_batch,
                                                        number_of_indices_t target_indices_per_host_batch,
                                                        number_of_indices_t target_indices_per_device_batch,
                                                        const std::shared_ptr<const claragenomics::io::FastaParser> query_parser,
                                                        const std::shared_ptr<const claragenomics::io::FastaParser> target_parser,
                                                        number_of_basepairs_t query_basepairs_per_index,
                                                        number_of_basepairs_t target_basepairs_per_index,
                                                        bool same_query_and_target);

namespace details
{
namespace index_batcher
{

/// \brief groups query and target IndexDescriptors into batches
///
/// see generate_batches_of_indices() for more details
///
/// \param query_indices
/// \param target_indices
/// \param query_indices_per_batch
/// \param target_indices_per_batch
/// \param same_query_and_target
/// \return generated batches
std::vector<IndexBatch> group_into_batches(const std::vector<IndexDescriptor>& query_indices,
                                           const std::vector<IndexDescriptor>& target_indices,
                                           number_of_indices_t query_indices_per_batch,
                                           number_of_indices_t target_indices_per_batch,
                                           bool same_query_and_target);

} // namespace index_batcher
} // namespace details

} // namespace cudamapper
} // namespace claragenomics
