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

DeviceIndexCache::DeviceIndexCache(const CacheType cache_type,
                                   HostIndexCache* host_cache)
    : cache_type_(cache_type)
    , host_cache_(host_cache)
    , is_ready_(false)
{
    host_cache_->register_device_cache(cache_type_,
                                       this);
}

DeviceIndexCache::~DeviceIndexCache()
{
    host_cache_->deregister_device_cache(cache_type_,
                                         this);
}

void DeviceIndexCache::add_index(const IndexDescriptor index_descriptor,
                                 std::shared_ptr<Index> device_index)
{
    cache_[index_descriptor] = device_index;
}

std::shared_ptr<Index> DeviceIndexCache::get_index(const IndexDescriptor index_descriptor) const
{
    if (!is_ready())
    {
        throw DeviceCacheNotReadyException(cache_type_,
                                           index_descriptor);
    }

    return get_index_no_check_if_ready(index_descriptor);
}

std::shared_ptr<Index> DeviceIndexCache::get_index_no_check_if_ready(IndexDescriptor index_descriptor) const
{
    const auto index_iter = cache_.find(index_descriptor);
    if (index_iter == cache_.end())
    {
        throw IndexNotFoundException(cache_type_,
                                     IndexNotFoundException::IndexLocation::device_cache,
                                     index_descriptor);
    }
    return index_iter->second;
}

bool DeviceIndexCache::has_index(const IndexDescriptor index_descriptor) const
{
    // TODO: use optional instead
    return 0 != cache_.count(index_descriptor);
}

void DeviceIndexCache::wait_for_data_to_be_ready()
{
    if (!is_ready())
    {
        for (const auto& index_it : cache_)
        {
            index_it.second->wait_to_be_ready();
        }

        is_ready_ = true;
    }
}

bool DeviceIndexCache::is_ready() const
{
    return is_ready_;
}

HostIndexCache::HostIndexCache(const bool same_query_and_target,
                               genomeworks::DefaultDeviceAllocator allocator,
                               std::shared_ptr<genomeworks::io::FastaParser> query_parser,
                               std::shared_ptr<genomeworks::io::FastaParser> target_parser,
                               const std::uint64_t kmer_size,
                               const std::uint64_t window_size,
                               const bool hash_representations,
                               const double filtering_parameter,
                               cudaStream_t cuda_stream_generation,
                               cudaStream_t cuda_stream_copy)
    : same_query_and_target_(same_query_and_target)
    , allocator_(allocator)
    , query_parser_(query_parser)
    , target_parser_(target_parser)
    , kmer_size_(kmer_size)
    , window_size_(window_size)
    , hash_representations_(hash_representations)
    , filtering_parameter_(filtering_parameter)
    , cuda_stream_generation_(cuda_stream_generation)
    , cuda_stream_copy_(cuda_stream_copy)
{
}

void HostIndexCache::generate_content(const CacheType cache_type,
                                      const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                      const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                      const bool skip_copy_to_host)
{
    // skip_copy_to_host only makes sense if descriptors_of_indices_to_cache and descriptors_of_indices_to_keep_on_device are the same
    // otherwise some indices would be created and not saved on either host or device
    assert(!skip_copy_to_host || (descriptors_of_indices_to_cache == descriptors_of_indices_to_keep_on_device));
    assert(!descriptors_of_indices_to_cache.empty());

    host_cache_t& this_cache                   = (CacheType::query_cache == cache_type) ? query_host_cache_ : target_host_cache_;
    const host_cache_t& other_cache            = (CacheType::query_cache == cache_type) ? target_host_cache_ : query_host_cache_;
    device_cache_t& indices_kept_on_device     = (CacheType::query_cache == cache_type) ? query_indices_kept_on_device_ : target_indices_kept_on_device_;
    const genomeworks::io::FastaParser* parser = (CacheType::query_cache == cache_type) ? query_parser_.get() : target_parser_.get();

    // convert descriptors_of_indices_to_keep_on_device into set for faster search
    std::unordered_set<IndexDescriptor, IndexDescriptorHash> descriptors_of_indices_to_keep_on_device_set(begin(descriptors_of_indices_to_keep_on_device),
                                                                                                          end(descriptors_of_indices_to_keep_on_device));

    host_cache_t new_cache;
    indices_kept_on_device.clear(); // normally this should be empty anyway

    // In most cases index is generated on device and then moved to host. These two operations can be overlapped, i.e. while one index is being copied
    // to host the next index can be generated.
    // Index cache is expected to be larger than the available device memory, meaning it is not possible to keep all indices on device while they are
    // being copied to host. In this implementation only two copies of index are kept on device: the one currently being generated and the one currently
    // being copied to host (from the previous step).

    std::shared_ptr<const IndexHostCopyBase> index_on_host = nullptr;
    std::shared_ptr<Index> index_on_device                 = nullptr;
    // Only one pair of (host, device) indices can be copied at a time. Whenever a copy should be started if these pointers are not null (= copy is in progress)
    // wait for that current copy to be done first
    std::shared_ptr<const IndexHostCopyBase> index_on_host_copy_in_flight = nullptr;
    std::shared_ptr<Index> index_on_device_copy_in_flight                 = nullptr;

    const bool host_copy_needed = !skip_copy_to_host;

    for (const IndexDescriptor& descriptor_of_index_to_cache : descriptors_of_indices_to_cache)
    {
        index_on_host   = nullptr;
        index_on_device = nullptr;

        const bool device_copy_needed = descriptors_of_indices_to_keep_on_device_set.count(descriptor_of_index_to_cache) != 0;

        // check if host copy already exists

        // check if index is already in this cache
        auto index_in_this_cache = this_cache.find(descriptor_of_index_to_cache);
        if (index_in_this_cache != this_cache.end())
        {
            // index already cached
            index_on_host = index_in_this_cache->second;
        }
        // if index not found in this cache and query and target input files are the same check the other cache as well
        if (!index_on_host && same_query_and_target_)
        {
            auto index_in_other_cache = other_cache.find(descriptor_of_index_to_cache);
            if (index_in_other_cache != other_cache.end())
            {
                index_on_host = index_in_other_cache->second;
            }
        }

        if (!index_on_host)
        {
            // create index
            index_on_device = Index::create_index_async(allocator_,
                                                        *parser,
                                                        descriptor_of_index_to_cache,
                                                        kmer_size_,
                                                        window_size_,
                                                        hash_representations_,
                                                        filtering_parameter_,
                                                        cuda_stream_generation_,
                                                        cuda_stream_copy_);
            index_on_device->wait_to_be_ready();

            if (host_copy_needed)
            {
                // if a copy is already in progress wait for it to finish
                if (index_on_host_copy_in_flight)
                {
                    assert(index_on_host_copy_in_flight && index_on_device_copy_in_flight);
                    index_on_host_copy_in_flight->finish_copying();
                }

                index_on_host = IndexHostCopy::create_host_copy_async(*index_on_device,
                                                                      descriptor_of_index_to_cache.first_read(),
                                                                      kmer_size_,
                                                                      window_size_,
                                                                      cuda_stream_copy_);

                index_on_host_copy_in_flight   = index_on_host;
                index_on_device_copy_in_flight = index_on_device;
            }
        }

        if (index_on_host)
        {
            new_cache[descriptor_of_index_to_cache] = index_on_host;
        }

        // Device copy of index is only saved if is already exists, i.e. if the index has been generated
        // If the index has been found on host it won't be copied back to device at this point
        // TODO: check whether this index is already present in device cache, this is not expected to happen frequently so performance gains are going to be small
        if (device_copy_needed && index_on_device)
        {
            indices_kept_on_device[descriptor_of_index_to_cache] = index_on_device;
        }
    }

    // wait for the last copy to finish
    if (index_on_host_copy_in_flight)
    {
        assert(index_on_host_copy_in_flight && index_on_device_copy_in_flight);
        index_on_host_copy_in_flight->finish_copying();
    }

    std::swap(new_cache, this_cache);
}

std::shared_ptr<DeviceIndexCache> HostIndexCache::start_copying_indices_to_device(const CacheType cache_type,
                                                                                  const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    const host_cache_t& host_cache                            = (CacheType::query_cache == cache_type) ? query_host_cache_ : target_host_cache_;
    const std::vector<DeviceIndexCache*>& this_device_caches  = (CacheType::query_cache == cache_type) ? query_device_caches_ : target_device_caches_;
    const std::vector<DeviceIndexCache*>& other_device_caches = (CacheType::query_cache == cache_type) ? target_device_caches_ : query_device_caches_;
    device_cache_t& indices_kept_on_device                    = (CacheType::query_cache == cache_type) ? query_indices_kept_on_device_ : target_indices_kept_on_device_;

    std::shared_ptr<DeviceIndexCache> device_cache = std::make_shared<DeviceIndexCache>(cache_type,
                                                                                        this);

    for (const IndexDescriptor& descriptor : descriptors_of_indices_to_cache)
    {
        std::shared_ptr<Index> device_index = nullptr;

        // check if index was kept on device after creation
        auto index_kept_on_device = indices_kept_on_device.find(descriptor);
        if (index_kept_on_device != indices_kept_on_device.end())
        {
            device_index = index_kept_on_device->second;
            indices_kept_on_device.erase(descriptor);
        }

        // check if value is already in this cache
        if (!device_index)
        {
            for (const DeviceIndexCache* const existing_cache : this_device_caches)
            {
                if (existing_cache != device_cache.get() && existing_cache->has_index(descriptor))
                {
                    device_index = existing_cache->get_index_no_check_if_ready(descriptor);
                    break;
                }
            }
        }

        // if query and target files are the same check the other index as well
        if (!device_index && same_query_and_target_)
        {
            for (const DeviceIndexCache* const existing_cache : other_device_caches)
            {
                if (existing_cache->has_index(descriptor))
                {
                    device_index = existing_cache->get_index_no_check_if_ready(descriptor);
                    break;
                }
            }
        }

        // if index has not been found on device copy it from host
        if (!device_index)
        {
            const auto index_on_host_iter = host_cache.find(descriptor);
            if (index_on_host_iter == host_cache.end())
            {
                throw IndexNotFoundException(cache_type,
                                             IndexNotFoundException::IndexLocation::host_cache,
                                             descriptor);
            }
            device_index = index_on_host_iter->second->copy_index_to_device(allocator_, cuda_stream_copy_);
        }

        assert(device_index);
        device_cache->add_index(descriptor,
                                device_index);
    }

    return device_cache;
}

void HostIndexCache::register_device_cache(const CacheType cache_type,
                                           DeviceIndexCache* index_cache)
{
    assert(cache_type == CacheType::query_cache || cache_type == CacheType::target_cache);

    std::vector<DeviceIndexCache*>& device_caches = cache_type == CacheType::query_cache ? query_device_caches_ : target_device_caches_;

    device_caches.push_back(index_cache);
}

void HostIndexCache::deregister_device_cache(const CacheType cache_type,
                                             DeviceIndexCache* index_cache)
{
    assert(cache_type == CacheType::query_cache || cache_type == CacheType::target_cache);

    std::vector<DeviceIndexCache*>& device_caches = cache_type == CacheType::query_cache ? query_device_caches_ : target_device_caches_;

    auto new_end = std::remove(begin(device_caches), end(device_caches), index_cache);
    device_caches.erase(new_end, end(device_caches));
}

IndexNotFoundException::IndexNotFoundException(const CacheType cache_type,
                                               const IndexLocation index_location,
                                               const IndexDescriptor index_descriptor)
    : message_(std::string(cache_type == CacheType::query_cache ? "Query " : "Target ") +
               "index not found in " +
               std::string(index_location == IndexLocation::host_cache ? "host " : "device ") +
               "cache. First read: " +
               std::to_string(index_descriptor.first_read()) +
               ", number of reads: " +
               std::to_string(index_descriptor.number_of_reads()))
{
}

const char* IndexNotFoundException::what() const noexcept
{
    return message_.c_str();
}

DeviceCacheNotReadyException::DeviceCacheNotReadyException(const CacheType cache_type,
                                                           const IndexDescriptor index_descriptor)
    : message_("Cache for " +
               std::string(cache_type == CacheType::query_cache ? "query " : "target ") +
               "index is not ready. First read: " +
               std::to_string(index_descriptor.first_read()) +
               ", number of reads: " +
               std::to_string(index_descriptor.number_of_reads()))
{
}

const char* DeviceCacheNotReadyException::what() const noexcept
{
    return message_.c_str();
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
