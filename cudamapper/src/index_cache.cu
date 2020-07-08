/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "index_cache.cuh"

#include "index_host_copy.cuh"

#include <unordered_set>

#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

IndexCacheHost::IndexCacheHost(const bool same_query_and_target,
                               genomeworks::DefaultDeviceAllocator allocator,
                               std::shared_ptr<genomeworks::io::FastaParser> query_parser,
                               std::shared_ptr<genomeworks::io::FastaParser> target_parser,
                               const std::uint64_t kmer_size,
                               const std::uint64_t window_size,
                               const bool hash_representations,
                               const double filtering_parameter,
                               const cudaStream_t cuda_stream)
    : same_query_and_target_(same_query_and_target)
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

void IndexCacheHost::generate_query_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                                  const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                                  const bool skip_copy_to_host)
{
    generate_cache_content(descriptors_of_indices_to_cache,
                           descriptors_of_indices_to_keep_on_device,
                           skip_copy_to_host,
                           CacheSelector::query_cache);
}

void IndexCacheHost::generate_target_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                                   const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                                   const bool skip_copy_to_host)
{
    generate_cache_content(descriptors_of_indices_to_cache,
                           descriptors_of_indices_to_keep_on_device,
                           skip_copy_to_host,
                           CacheSelector::target_cache);
}

std::shared_ptr<Index> IndexCacheHost::get_index_from_query_cache(const IndexDescriptor& descriptor_of_index_to_cache)
{
    return get_index_from_cache(descriptor_of_index_to_cache,
                                CacheSelector::query_cache);
}

std::shared_ptr<Index> IndexCacheHost::get_index_from_target_cache(const IndexDescriptor& descriptor_of_index_to_cache)
{
    return get_index_from_cache(descriptor_of_index_to_cache,
                                CacheSelector::target_cache);
}

void IndexCacheHost::generate_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                            const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                            const bool skip_copy_to_host,
                                            const CacheSelector which_cache)
{
    // skip_copy_to_host only makes sense if descriptors_of_indices_to_cache and descriptors_of_indices_to_keep_on_device are the same
    // otherwise some indices would be created and not saved on either host or device
    assert(!skip_copy_to_host || (descriptors_of_indices_to_cache == descriptors_of_indices_to_keep_on_device));

    cache_type_t& cache_to_edit                           = (CacheSelector::query_cache == which_cache) ? query_cache_ : target_cache_;
    const cache_type_t& cache_to_check                    = (CacheSelector::query_cache == which_cache) ? target_cache_ : query_cache_;
    device_cache_type_t& temp_device_cache_to_edit        = (CacheSelector::query_cache == which_cache) ? query_temp_device_cache_ : target_temp_device_cache_;
    const device_cache_type_t& temp_device_cache_to_check = (CacheSelector::query_cache == which_cache) ? target_temp_device_cache_ : query_temp_device_cache_;
    const genomeworks::io::FastaParser* parser            = (CacheSelector::query_cache == which_cache) ? query_parser_.get() : target_parser_.get();

    // convert descriptors_of_indices_to_keep_on_device into set for faster search
    std::unordered_set<IndexDescriptor, IndexDescriptorHash> descriptors_of_indices_to_keep_on_device_set(begin(descriptors_of_indices_to_keep_on_device),
                                                                                                          end(descriptors_of_indices_to_keep_on_device));

    cache_type_t new_cache;
    temp_device_cache_to_edit.clear(); // this should be empty by now anyway

    for (const IndexDescriptor& descriptor_of_index_to_cache : descriptors_of_indices_to_cache)
    {
        // check if this index should be kept on device in addition to copying it to host
        const bool keep_on_device = descriptors_of_indices_to_keep_on_device_set.count(descriptor_of_index_to_cache) != 0;

        std::shared_ptr<const IndexHostCopyBase> index_on_host = nullptr;
        std::shared_ptr<Index> index_on_device                 = nullptr;

        if (same_query_and_target_)
        {
            // check if the same index already exists in the other cache
            auto existing_cache = cache_to_check.find(descriptor_of_index_to_cache);
            if (existing_cache != cache_to_check.end())
            {
                index_on_host = existing_cache->second;
                if (keep_on_device)
                {
                    auto existing_device_cache = temp_device_cache_to_check.find(descriptor_of_index_to_cache);
                    if (existing_device_cache != temp_device_cache_to_check.end())
                    {
                        index_on_device = existing_device_cache->second;
                    }
                    else
                    {
                        index_on_device = index_on_host->copy_index_to_device(allocator_, cuda_stream_);
                    }
                }
            }
        }

        if (nullptr == index_on_host)
        {
            // check if this index is already cached in this cache
            auto existing_cache = cache_to_edit.find(descriptor_of_index_to_cache);
            if (existing_cache != cache_to_edit.end())
            {
                // index already cached
                index_on_host = existing_cache->second;
                if (keep_on_device)
                {
                    index_on_device = index_on_host->copy_index_to_device(allocator_, cuda_stream_);
                }
            }
            else
            {
                // create index
                index_on_device = Index::create_index(allocator_,
                                                      *parser,
                                                      descriptor_of_index_to_cache.first_read(),
                                                      descriptor_of_index_to_cache.first_read() + descriptor_of_index_to_cache.number_of_reads(),
                                                      kmer_size_,
                                                      window_size_,
                                                      hash_representations_,
                                                      filtering_parameter_,
                                                      cuda_stream_);
                // copy it to host memory
                if (!skip_copy_to_host)
                {
                    index_on_host = IndexHostCopy::create_cache(*index_on_device,
                                                                descriptor_of_index_to_cache.first_read(),
                                                                kmer_size_,
                                                                window_size_,
                                                                cuda_stream_);
                }
            }
        }

        // save pointer to cached index
        if (!skip_copy_to_host)
        {
            assert(nullptr != index_on_host);
            new_cache[descriptor_of_index_to_cache] = index_on_host;
        }

        if (keep_on_device)
        {
            temp_device_cache_to_edit[descriptor_of_index_to_cache] = index_on_device;
        }
    }

    std::swap(new_cache, cache_to_edit);
}

std::shared_ptr<Index> IndexCacheHost::get_index_from_cache(const IndexDescriptor& descriptor_of_index_to_cache,
                                                            const CacheSelector which_cache)
{
    std::shared_ptr<Index> index;

    const cache_type_t& host_cache               = (CacheSelector::query_cache == which_cache) ? query_cache_ : target_cache_;
    device_cache_type_t& temp_device_index_cache = (CacheSelector::query_cache == which_cache) ? query_temp_device_cache_ : target_temp_device_cache_;

    auto temp_device_index_cache_iter = temp_device_index_cache.find(descriptor_of_index_to_cache);
    // check if index is present in device memory, copy from host if not
    if (temp_device_index_cache_iter != temp_device_index_cache.end())
    {
        index = temp_device_index_cache_iter->second;
        // indices are removed from device cache after they have been used for the first time
        temp_device_index_cache.erase(temp_device_index_cache_iter);
    }
    else
    {
        // TODO: throw custom exception if index not found
        index = host_cache.at(descriptor_of_index_to_cache)->copy_index_to_device(allocator_, cuda_stream_);
    }

    return index;
}

IndexCacheDevice::IndexCacheDevice(const bool same_query_and_target,
                                   std::shared_ptr<IndexCacheHost> index_cache_host)
    : same_query_and_target_(same_query_and_target)
    , index_cache_host_(index_cache_host)
{
}

void IndexCacheDevice::generate_query_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    generate_cache_content(descriptors_of_indices_to_cache, CacheSelector::query_cache);
}

void IndexCacheDevice::generate_target_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    generate_cache_content(descriptors_of_indices_to_cache, CacheSelector::target_cache);
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

void IndexCacheDevice::generate_cache_content(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                              const CacheSelector which_cache)
{
    cache_type_t& cache_to_edit        = (CacheSelector::query_cache == which_cache) ? query_cache_ : target_cache_;
    const cache_type_t& cache_to_check = (CacheSelector::query_cache == which_cache) ? target_cache_ : query_cache_;

    cache_type_t new_cache;

    for (const IndexDescriptor& descriptor_of_index_to_cache : descriptors_of_indices_to_cache)
    {

        std::shared_ptr<Index> index = nullptr;

        if (same_query_and_target_)
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
            // check if this index is already cached in this cache
            auto existing_cache = cache_to_edit.find(descriptor_of_index_to_cache);
            if (existing_cache != cache_to_edit.end())
            {
                // index already cached
                index = existing_cache->second;
            }
            else
            {
                // index not already cached -> fetch it from index_cache_host_
                if (CacheSelector::query_cache == which_cache)
                {
                    index = index_cache_host_->get_index_from_query_cache(descriptor_of_index_to_cache);
                }
                else
                {
                    index = index_cache_host_->get_index_from_target_cache(descriptor_of_index_to_cache);
                }
            }
        }

        assert(nullptr != index);

        // save pointer to cached index
        new_cache[descriptor_of_index_to_cache] = index;
    }

    std::swap(new_cache, cache_to_edit);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
