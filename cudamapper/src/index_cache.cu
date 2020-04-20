/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "index_cache.cuh"

#include "index_host_copy.cuh"

#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/io/fasta_parser.hpp>

namespace claragenomics
{
namespace cudamapper
{

IndexCacheHost::IndexCacheHost(const bool reuse_data,
                               claragenomics::DefaultDeviceAllocator allocator,
                               std::shared_ptr<claragenomics::io::FastaParser> query_parser,
                               std::shared_ptr<claragenomics::io::FastaParser> target_parser,
                               const std::uint64_t kmer_size,
                               const std::uint64_t window_size,
                               const bool hash_representations,
                               const double filtering_parameter,
                               const cudaStream_t cuda_stream)
    : reuse_data_(reuse_data)
    , allocator_(allocator)
    , query_parser_(query_parser)
    , target_parser_(target_parser)
    , kmer_size_(kmer_size)
    , window_size_(window_size)
    , hash_representations_(hash_representations)
    , filtering_parameter_(filtering_parameter)
    , cuda_stream_(cuda_stream)
{
}

void IndexCacheHost::update_query_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    update_cache(descriptors_of_indices_to_cache, CacheToUpdate::query);
}

void IndexCacheHost::update_target_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    update_cache(descriptors_of_indices_to_cache, CacheToUpdate::target);
}

std::shared_ptr<Index> IndexCacheHost::get_index_from_query_cache(const IndexDescriptor& descriptor_of_index_to_cache)
{
    // TODO: throw custom exception if index not found
    return query_cache_.at(descriptor_of_index_to_cache)->copy_index_to_device(allocator_, cuda_stream_);
}

std::shared_ptr<Index> IndexCacheHost::get_index_from_target_cache(const IndexDescriptor& descriptor_of_index_to_cache)
{
    // TODO: throw custom exception if index not found
    return target_cache_.at(descriptor_of_index_to_cache)->copy_index_to_device(allocator_, cuda_stream_);
}

void IndexCacheHost::update_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                  const CacheToUpdate which_cache)
{
    cache_type_t& cache_to_edit                  = (CacheToUpdate::query == which_cache) ? query_cache_ : target_cache_;
    const cache_type_t& cache_to_check           = (CacheToUpdate::query == which_cache) ? target_cache_ : query_cache_;
    const claragenomics::io::FastaParser* parser = (CacheToUpdate::query == which_cache) ? query_parser_.get() : target_parser_.get();

    cache_type_t new_cache;

    for (const IndexDescriptor& descriptor_of_index_to_cache : descriptors_of_indices_to_cache)
    {

        std::shared_ptr<const IndexHostCopyBase> index_copy = nullptr;

        if (reuse_data_)
        {
            // check if the same index already exists in the other cache
            auto existing_cache = cache_to_check.find(descriptor_of_index_to_cache);
            if (existing_cache != cache_to_check.end())
            {
                index_copy = existing_cache->second;
            }
        }

        if (nullptr == index_copy)
        {
            // create index
            auto index = claragenomics::cudamapper::Index::create_index(allocator_,
                                                                        *parser,
                                                                        descriptor_of_index_to_cache.first_read(),
                                                                        descriptor_of_index_to_cache.first_read() + descriptor_of_index_to_cache.number_of_reads(),
                                                                        kmer_size_,
                                                                        window_size_,
                                                                        hash_representations_,
                                                                        filtering_parameter_,
                                                                        cuda_stream_);
            // copy it to host memory
            index_copy = claragenomics::cudamapper::IndexHostCopy::create_cache(*index,
                                                                                descriptor_of_index_to_cache.first_read(),
                                                                                kmer_size_,
                                                                                window_size_,
                                                                                cuda_stream_);
        }

        assert(nullptr != index_copy);

        // save pointer to cached index
        new_cache[descriptor_of_index_to_cache] = index_copy;
    }

    std::swap(new_cache, cache_to_edit);
}

IndexCacheDevice::IndexCacheDevice(const bool reuse_data,
                                   std::shared_ptr<IndexCacheHost> index_cache_host)
    : reuse_data_(reuse_data)
    , index_cache_host_(index_cache_host)
{
}

void IndexCacheDevice::update_query_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    update_cache(descriptors_of_indices_to_cache, CacheToUpdate::query);
}

void IndexCacheDevice::update_target_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    update_cache(descriptors_of_indices_to_cache, CacheToUpdate::target);
}

std::shared_ptr<Index> IndexCacheDevice::get_index_from_query_cache(const IndexDescriptor& descriptor_of_index_to_cache)
{
    // TODO: throw custom exception if index not found
    return query_cache_.at(descriptor_of_index_to_cache);
}

std::shared_ptr<Index> IndexCacheDevice::get_index_from_target_cache(const IndexDescriptor& descriptor_of_index_to_cache)
{
    // TODO: throw custom exception if index not found
    return target_cache_.at(descriptor_of_index_to_cache);
}

void IndexCacheDevice::update_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                    const CacheToUpdate which_cache)
{
    cache_type_t& cache_to_edit        = (CacheToUpdate::query == which_cache) ? query_cache_ : target_cache_;
    const cache_type_t& cache_to_check = (CacheToUpdate::query == which_cache) ? target_cache_ : query_cache_;

    cache_type_t new_cache;

    for (const IndexDescriptor& descriptor_of_index_to_cache : descriptors_of_indices_to_cache)
    {

        std::shared_ptr<Index> index = nullptr;

        if (reuse_data_)
        {
            // check if the same index already exists in the other cache
            auto existing_cache = cache_to_check.find(descriptor_of_index_to_cache);
            if (existing_cache != cache_to_check.end())
            {
                index = existing_cache->second;
            }
        }

        if (nullptr == index)
        {
            if (CacheToUpdate::query == which_cache)
            {
                index = index_cache_host_->get_index_from_query_cache(descriptor_of_index_to_cache);
            }
            else
            {
                index = index_cache_host_->get_index_from_target_cache(descriptor_of_index_to_cache);
            }
        }

        assert(nullptr != index);

        // save pointer to cached index
        new_cache[descriptor_of_index_to_cache] = index;
    }

    std::swap(new_cache, cache_to_edit);
}

} // namespace cudamapper
} // namespace claragenomics
