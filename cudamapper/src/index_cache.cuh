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

#include <memory>
#include <unordered_map>

#include <claragenomics/cudamapper/types.hpp>
#include <claragenomics/utils/allocator.hpp>

#include "index_descriptor.hpp"

namespace claragenomics
{

namespace io
{
class FastaParser;
} // namespace io

namespace cudamapper
{

class Index;
class IndexHostCopyBase;

/// IndexCacheHost - Creates Indices, stores them in host memory and on demand copies them back to device memory
///
/// The user tells cache which Indices to keep in cache using generate_query_cache_content() and generate_target_cache_content() and
/// retrieves indices using get_index_from_query_cache() and get_index_from_target_cache(). Trying to retrieve an
/// Index which was not previously stored in cache results in an exception
class IndexCacheHost
{
public:
    /// \brief Constructor
    /// \param same_query_and_target true means that both query and target are the same, meaning that if requested index exists in query cache it can also be used by target cache directly
    /// \param allocator allocator to use for device arrays
    /// \param query_parser
    /// \param target_parser
    /// \param kmer_size // see Index
    /// \param window_size // see Index
    /// \param hash_representations // see Index
    /// \param filtering_parameter // see Index
    /// \param cuda_stream // device memory used for Index copy will only we freed up once all previously scheduled work on this stream has finished
    IndexCacheHost(bool same_query_and_target,
                   claragenomics::DefaultDeviceAllocator allocator,
                   std::shared_ptr<claragenomics::io::FastaParser> query_parser,
                   std::shared_ptr<claragenomics::io::FastaParser> target_parser,
                   std::uint64_t kmer_size,
                   std::uint64_t window_size,
                   bool hash_representations  = true,
                   double filtering_parameter = 1.0,
                   cudaStream_t cuda_stream   = 0);

    IndexCacheHost(const IndexCacheHost&) = delete;
    IndexCacheHost& operator=(const IndexCacheHost&) = delete;
    IndexCacheHost(IndexCacheHost&&)                 = delete;
    IndexCacheHost& operator=(IndexCacheHost&&) = delete;
    ~IndexCacheHost()                           = default;

    /// \brief Discards previously cached query Indices, creates new Indices and copies them to host memory
    ///
    /// Expected usage pattern is to call generate_query_cache_content() with certain index descriptors and then immediately retrieve some of them.
    /// To avoid immediately copying back those indices from host to device it is possible to specify descriptors_of_indices_to_keep_on_device
    //  which will be copied to host, but also kept on device until retrieved for the first time using get_index_from_query_cache
    ///
    /// \param descriptors_of_indices_to_cache descriptors on indices to keep in host memory
    /// \param descriptors_of_indices_to_keep_on_device descriptors of indices to keep in device memory in addition to host memory until retrieved for the first time
    void generate_query_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                      const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device = {});

    /// \brief Discards previously cached target Indices, creates new Indices and copies them to host memory
    ///
    /// Expected usage pattern is to call generate_target_cache_content() with certain index descriptors and then immediately retrieve some of them.
    /// To avoid immediately copying back those indices from host to device it is possible to specify descriptors_of_indices_to_keep_on_device
    /// which will be copied to host, but also kept on device until retrieved for the first time using get_index_from_taget_cache
    ///
    /// \param descriptors_of_indices_to_cache descriptors on indices to keep in host memory
    /// \param descriptors_of_indices_to_keep_on_device descriptors of indices to keep in device memory in addition to host memory until retrieved for the first time
    void generate_target_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                       const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device = {});

    /// \brief Copies request Index to device memory
    /// throws if that index is currently not in cache
    std::shared_ptr<Index> get_index_from_query_cache(const IndexDescriptor& descriptor_of_index_to_cache);

    /// \brief Copies request Index to device memory
    /// throws if that index is currently not in cache
    std::shared_ptr<Index> get_index_from_target_cache(const IndexDescriptor& descriptor_of_index_to_cache);

private:
    using cache_type_t = std::unordered_map<IndexDescriptor,
                                            std::shared_ptr<const IndexHostCopyBase>,
                                            IndexDescriptorHash>;

    using device_cache_type_t = std::unordered_map<IndexDescriptor,
                                                   std::shared_ptr<Index>,
                                                   IndexDescriptorHash>;

    enum class CacheSelector
    {
        query_cache,
        target_cache
    };

    /// \brief Discards previously cached Indices, creates new Indices and copies them to host memory
    /// Uses which_cache to determine if it should be working on query of target indices
    ///
    /// If same_query_and_target_ is true function checks the other cache to see if that index is already in cache
    void generate_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                CacheSelector which_cache);

    /// \brief Fetches requested index
    /// Copies index from host to device memory, unless index is saved in temp device cache
    /// If that is the case it returs that device copy are removes it from temp device cache
    std::shared_ptr<Index> get_index_from_cache(const IndexDescriptor& descriptor_of_index_to_cache,
                                                CacheSelector which_cache);

    /// Host copies of indices
    cache_type_t query_cache_;
    cache_type_t target_cache_;
    /// User can instruct cache to also keep certain indices in device memory until retrieved for the first time
    device_cache_type_t query_temp_device_cache_;
    device_cache_type_t target_temp_device_cache_;

    const bool same_query_and_target_;
    claragenomics::DefaultDeviceAllocator allocator_;
    std::shared_ptr<claragenomics::io::FastaParser> query_parser_;
    std::shared_ptr<claragenomics::io::FastaParser> target_parser_;
    const std::uint64_t kmer_size_;
    const std::uint64_t window_size_;
    const bool hash_representations_;
    const double filtering_parameter_;
    const cudaStream_t cuda_stream_;
};

/// IndexCacheDevice - Keeps copies of Indices in device memory
///
/// The user tells cache which Indices to keep in cache using generate_query_cache_content() and generate_target_cache_content() and
/// retrieves indices using get_index_from_query_cache() and get_index_from_target_cache(). Trying to retrieve an
/// Index which was not previously stored in cache results in an exception.
///
/// IndexCacheDevice relies on IndexCacheHost to provide actual indices for caching
class IndexCacheDevice
{
public:
    /// \brief Constructor
    /// \param same_query_and_target true means that both query and target are the same, meaning that if requested index exists in query cache it can also be used by target cache directly
    /// \param index_cache_host underlying host cache to get the indices from
    IndexCacheDevice(bool same_query_and_target,
                     std::shared_ptr<IndexCacheHost> index_cache_host);

    IndexCacheDevice(const IndexCacheDevice&) = delete;
    IndexCacheDevice& operator=(const IndexCacheDevice&) = delete;
    IndexCacheDevice(IndexCacheDevice&&)                 = delete;
    IndexCacheDevice& operator=(IndexCacheDevice&&) = delete;
    ~IndexCacheDevice()                             = default;

    /// \brief Discards previously cached query Indices, creates new Indices and copies them to host memory
    void generate_query_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache);

    /// \brief Discards previously cached target Indices, creates new Indices and copies them to host memory
    void generate_target_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache);

    /// \brief Copies request Index to device memory
    /// throws if that index is currently not in cache
    std::shared_ptr<Index> get_index_from_query_cache(const IndexDescriptor& descriptor_of_index_to_cache);

    /// \brief Copies request Index to device memory
    /// throws if that index is currently not in cache
    std::shared_ptr<Index> get_index_from_target_cache(const IndexDescriptor& descriptor_of_index_to_cache);

private:
    using cache_type_t = std::unordered_map<IndexDescriptor,
                                            std::shared_ptr<Index>,
                                            IndexDescriptorHash>;

    enum class CacheSelector
    {
        query_cache,
        target_cache
    };

    /// \brief Discards previously cached Indices and caches new ones
    /// Uses which_cache to determine if it should be working on query of target indices
    ///
    /// If same_query_and_target_ is true function checks the other cache to see if that index is already in cache
    void generate_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                CacheSelector which_cache);

    cache_type_t query_cache_;
    cache_type_t target_cache_;

    const bool same_query_and_target_;
    std::shared_ptr<IndexCacheHost> index_cache_host_;
};

} // namespace cudamapper
} // namespace claragenomics
