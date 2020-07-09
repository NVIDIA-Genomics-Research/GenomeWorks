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

#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include <claraparabricks/genomeworks/utils/allocator.hpp>

#include "index_descriptor.hpp"

namespace claraparabricks
{

namespace genomeworks
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
    /// \param kmer_size see Index
    /// \param window_size see Index
    /// \param hash_representations see Index
    /// \param filtering_parameter see Index
    /// \param cuda_stream_generate index generation is done one this stream, device memory in resulting device copies of index will only we freed once all previously scheduled work on this stream has finished
    /// \param cuda_stream_copy D2H and H2D copies of indices will be done on this stra, device memory in resulting device copies of index will only we freed once all previously scheduled work on this stream has finished
    IndexCacheHost(bool same_query_and_target,
                   genomeworks::DefaultDeviceAllocator allocator,
                   std::shared_ptr<genomeworks::io::FastaParser> query_parser,
                   std::shared_ptr<genomeworks::io::FastaParser> target_parser,
                   std::uint64_t kmer_size,
                   std::uint64_t window_size,
                   bool hash_representations         = true,
                   double filtering_parameter        = 1.0,
                   cudaStream_t cuda_stream_generate = 0,
                   cudaStream_t cuda_stream_copy     = 0);

    IndexCacheHost(const IndexCacheHost&) = delete;
    IndexCacheHost& operator=(const IndexCacheHost&) = delete;
    IndexCacheHost(IndexCacheHost&&)                 = delete;
    IndexCacheHost& operator=(IndexCacheHost&&) = delete;
    ~IndexCacheHost()                           = default;

    /// \brief Discards previously cached query Indices, creates new Indices and starts copying them to host memory
    ///
    /// Copy is done asynchronously and one should wait for it to finish with finish_generating_query_cache_content()
    ///
    /// Expected usage pattern is to immediately after creation retrieve some of generated indices.
    /// To avoid immediately copying back those indices from host to device it is possible to specify descriptors_of_indices_to_keep_on_device
    //  which will be copied to host, but also kept on device until retrieved for the first time using get_index_from_query_cache
    ///
    /// \param descriptors_of_indices_to_cache descriptors on indices to keep in host memory
    /// \param descriptors_of_indices_to_keep_on_device descriptors of indices to keep in device memory in addition to host memory until retrieved for the first time
    /// \param skip_copy_to_host this option should be used if descriptors_of_indices_to_cache == descriptors_of_indices_to_keep_on_device and if each index will be queried only once, it's usefull for small cases where host cache isn't actually needed
    void start_generating_query_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                              const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device = {},
                                              bool skip_copy_to_host                                                       = false);

    /// \brief waits for copies started in start_generating_query_cache_content() to finish
    void finish_generating_query_cache_content();

    /// \brief Discards previously cached target Indices, creates new Indices and starts copying them to host memory
    ///
    /// Copy is done asynchronously and one should wait for it to finish with finish_generating_target_cache_content()
    ///
    /// Expected usage pattern is to immediately after creation retrieve some of generated indices.
    /// To avoid immediately copying back those indices from host to device it is possible to specify descriptors_of_indices_to_keep_on_device
    /// which will be copied to host, but also kept on device until retrieved for the first time using get_index_from_taget_cache
    ///
    /// \param descriptors_of_indices_to_cache descriptors on indices to keep in host memory
    /// \param descriptors_of_indices_to_keep_on_device descriptors of indices to keep in device memory in addition to host memory until retrieved for the first time
    /// \param skip_copy_to_host this option should be used if descriptors_of_indices_to_cache == descriptors_of_indices_to_keep_on_device and if each index will be queried only once, it's usefull for small cases where host cache isn't actually needed
    void start_generating_target_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                               const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device = {},
                                               bool skip_copy_to_host                                                       = false);

    /// \brief waits for copies started in start_generating_target_cache_content() to finish
    void finish_generating_target_cache_content();

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

    /// \brief Discards previously cached query Indices, creates new Indices and starts copying them to host memory
    ///
    /// Copy is done asynchronously and one should wait for it to finish with finish_generating_cache_content()
    ///
    /// Uses which_cache to determine if it should be working on query of target indices
    ///
    /// If same_query_and_target_ is true function checks the other cache to see if that index is already in cache
    void start_generating_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                        const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                        bool skip_copy_to_host,
                                        CacheSelector which_cache);

    /// \brief waits for copies started in start_generating_cache_content() to finish
    void finish_generating_cache_content(CacheSelector which_cache);

    /// \brief Fetches requested index
    /// Copies index from host to device memory, unless index is saved in temp device cache
    /// If that is the case it returs that device copy are removes it from temp device cache
    std::shared_ptr<Index> get_index_from_cache(const IndexDescriptor& descriptor_of_index_to_cache,
                                                CacheSelector which_cache);

    // Host copies of indices
    cache_type_t query_cache_;
    cache_type_t target_cache_;
    // User can instruct cache to also keep certain indices in device memory until retrieved for the first time
    device_cache_type_t query_temp_device_cache_;
    device_cache_type_t target_temp_device_cache_;

    // list of all indices whose generation or movement between host and device was started in start_generating_cache_content()
    std::vector<std::shared_ptr<const IndexHostCopyBase>> query_indices_in_progress;
    std::vector<std::shared_ptr<const IndexHostCopyBase>> target_indices_in_progress;

    const bool same_query_and_target_;
    genomeworks::DefaultDeviceAllocator allocator_;
    std::shared_ptr<genomeworks::io::FastaParser> query_parser_;
    std::shared_ptr<genomeworks::io::FastaParser> target_parser_;
    const std::uint64_t kmer_size_;
    const std::uint64_t window_size_;
    const bool hash_representations_;
    const double filtering_parameter_;
    const cudaStream_t cuda_stream_generation_;
    const cudaStream_t cuda_stream_copy_;
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

} // namespace genomeworks

} // namespace claraparabricks
