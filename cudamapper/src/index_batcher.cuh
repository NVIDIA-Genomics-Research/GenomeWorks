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
using number_of_indices_t = std::size_t;

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

/// index_id_t
using index_id_t = number_of_indices_t;

/// GroupOfIndicesDescriptor - describes a group of indices by the id of the first index and the total number of indices in that group
struct GroupOfIndicesDescriptor
{
    index_id_t first_index;
    number_of_indices_t number_of_indices;
};

/// GroupAndSubgroupsOfIndicesDescriptor - describes a group of indices and further divides it into subgroups
struct GroupAndSubgroupsOfIndicesDescriptor
{
    GroupOfIndicesDescriptor whole_group;
    std::vector<GroupOfIndicesDescriptor> subgroups;
};

/// HostAndDeviceGroupsOfIndices - holds all indices of host batch and device batches
struct HostAndDeviceGroupsOfIndices
{
    std::vector<IndexDescriptor> host_indices_group;
    std::vector<std::vector<IndexDescriptor>> device_indices_groups;
};

/// \brief Equality operator for HostAndDeviceGroupsOfIndices
bool operator==(const HostAndDeviceGroupsOfIndices& lhs, const HostAndDeviceGroupsOfIndices& rhs);

/// \brief Inequality operator for HostAndDeviceGroupsOfIndices
bool operator!=(const HostAndDeviceGroupsOfIndices& lhs, const HostAndDeviceGroupsOfIndices& rhs);

/// \brief Splits numbers into groups
///
/// Splits numbers between first_index and first_index + number_of_indices into groups of indices_per_group numbers.
/// If number_of_indices is not divisible by indices_per_group last group will have less elements.
///
/// For example for:
/// first_index = 16
/// number_of_indices = 25
/// indices_per_group = 7
/// generated groups will be:
/// (16, 7), (23, 7), (30, 7), (37, 4)
///
/// \param first_index
/// \param number_of_indices
/// \param indices_per_group
/// \return generated groups
std::vector<GroupOfIndicesDescriptor> split_array_into_groups(index_id_t first_index,
                                                              number_of_indices_t number_of_indices,
                                                              number_of_indices_t indices_per_group);

/// \brief Splits numbers into groups and then further splits each group into subgroups
///
/// Splits numbers between first_index and first_index + number_of_indices into groups of indices_per_group numbers.
/// If number_of_indices is not divisible by indices_per_group last group will have less elements.
/// After this it splits each group into subgroups of indices_per_subgroup elements.
///
/// For example for:
/// first_index = 100
/// number_of_indices = 71
/// indices_per_group = 16
/// indices_per_subgroup = 5
/// generated groups will be:
/// (100, 16) - (100, 5), (105, 5), (110, 5), (115, 1)
/// (116, 16) - (116, 5), (121, 5), (126, 5), (131, 1)
/// (132, 16) - (132, 5), (137, 5), (142, 5), (147, 1)
/// (148, 16) - (148, 5), (153, 5), (158, 5), (163, 1)
/// (164,  7) - (164, 5), (169, 2)
///
/// \param first_index
/// \param total_number_of_indices
/// \param indices_per_group
/// \param indices_per_subgroup
/// \return generated groups
std::vector<GroupAndSubgroupsOfIndicesDescriptor> generate_groups_and_subgroups(index_id_t first_index,
                                                                                number_of_indices_t total_number_of_indices,
                                                                                number_of_indices_t indices_per_group,
                                                                                number_of_indices_t indices_per_subgroup);

/// \brief Transforms descriptor of group of indices into descriptors of indices
/// \param index_descriptors descriptor of every individual index
/// \param groups_and_subgroups descriptors of groups of indices
/// \return groups of index descriptors
std::vector<HostAndDeviceGroupsOfIndices> convert_groups_of_indices_into_groups_of_index_descriptors(const std::vector<IndexDescriptor>& index_descriptors,
                                                                                                     const std::vector<GroupAndSubgroupsOfIndicesDescriptor>& groups_and_subgroups);

/// \brief Combines groups of query and target indices into batches of indices
///
/// See generate_batches_of_indices() for details
///
/// \param query_groups_of_indices
/// \param target_groups_of_indices
/// \param same_query_and_target
/// \return generated batches
std::vector<BatchOfIndices> combine_query_and_target_indices(const std::vector<HostAndDeviceGroupsOfIndices>& query_groups_of_indices,
                                                             const std::vector<HostAndDeviceGroupsOfIndices>& target_groups_of_indices,
                                                             bool same_query_and_target);

} // namespace index_batcher
} // namespace details

} // namespace cudamapper
} // namespace claragenomics
