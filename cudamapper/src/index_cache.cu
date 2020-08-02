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

IndexCache::IndexCache(bool same_query_and_target,
                       genomeworks::DefaultDeviceAllocator allocator,
                       std::shared_ptr<genomeworks::io::FastaParser> query_parser,
                       std::shared_ptr<genomeworks::io::FastaParser> target_parser,
                       std::uint64_t kmer_size,
                       std::uint64_t window_size,
                       bool hash_representations,
                       double filtering_parameter,
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

void IndexCache::generate_content_query_host(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                             const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                             const bool skip_copy_to_host)
{
    generate_content_host(descriptors_of_indices_to_cache,
                          descriptors_of_indices_to_keep_on_device,
                          skip_copy_to_host,
                          CacheSelector::query_cache);
}

void IndexCache::generate_content_target_host(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                              const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                              const bool skip_copy_to_host)
{
    generate_content_host(descriptors_of_indices_to_cache,
                          descriptors_of_indices_to_keep_on_device,
                          skip_copy_to_host,
                          CacheSelector::target_cache);
}

void IndexCache::start_generating_content_query_device(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    start_generating_content_device(descriptors_of_indices_to_cache,
                                    CacheSelector::query_cache);
}

void IndexCache::finish_generating_content_query_device()
{
    finish_generating_content_device(CacheSelector::query_cache);
}

void IndexCache::start_generating_content_target_device(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    start_generating_content_device(descriptors_of_indices_to_cache,
                                    CacheSelector::target_cache);
}

void IndexCache::finish_generating_content_target_device()
{
    finish_generating_content_device(CacheSelector::target_cache);
}

std::shared_ptr<const Index> IndexCache::get_index_from_query_cache(const IndexDescriptor& index_descriptor) const
{
    return get_index_from_cache(index_descriptor,
                                CacheSelector::query_cache);
}

std::shared_ptr<const Index> IndexCache::get_index_from_target_cache(const IndexDescriptor& index_descriptor) const
{
    return get_index_from_cache(index_descriptor,
                                CacheSelector::target_cache);
}

void IndexCache::generate_content_host(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                       const std::vector<IndexDescriptor>& descriptors_of_indices_to_keep_on_device,
                                       const bool skip_copy_to_host,
                                       const CacheSelector which_cache)
{
    // skip_copy_to_host only makes sense if descriptors_of_indices_to_cache and descriptors_of_indices_to_keep_on_device are the same
    // otherwise some indices would be created and not saved on either host or device
    assert(!skip_copy_to_host || (descriptors_of_indices_to_cache == descriptors_of_indices_to_keep_on_device));
    assert(!descriptors_of_indices_to_cache.empty());

    host_cache_t& this_cache                   = (CacheSelector::query_cache == which_cache) ? query_host_cache_ : target_host_cache_;
    const host_cache_t& other_cache            = (CacheSelector::query_cache == which_cache) ? target_host_cache_ : query_host_cache_;
    device_cache_t& indices_kept_on_device     = (CacheSelector::query_cache == which_cache) ? query_indices_kept_on_device_ : target_indices_kept_on_device_;
    const genomeworks::io::FastaParser* parser = (CacheSelector::query_cache == which_cache) ? query_parser_.get() : target_parser_.get();

    // convert descriptors_of_indices_to_keep_on_device into set for faster search
    std::unordered_set<IndexDescriptor, IndexDescriptorHash> descriptors_of_indices_to_keep_on_device_set(begin(descriptors_of_indices_to_keep_on_device),
                                                                                                          end(descriptors_of_indices_to_keep_on_device));

    host_cache_t new_cache;
    indices_kept_on_device.empty(); // normally this should be empty anyway

    // In most cases index is generated on device and then moved to host. These two operations can be overlapped, i.e. while one index is being copied
    // to host the next index can be generated.
    // Index cache is expected to be larger than the available device memory, meaning it is not possible to keep all indices on device while they are
    // being copied to host. In this implementation only two copies of index are kept on device: the one currently being generated and the one currently
    // being copied to host (from the previous step).

    std::shared_ptr<const IndexHostCopyBase> index_on_host                    = nullptr;
    std::shared_ptr<const IndexHostCopyBase> index_on_host_from_previous_step = nullptr;
    std::shared_ptr<Index> index_on_device                                    = nullptr;
    std::shared_ptr<Index> index_on_device_from_previous_step                 = nullptr;
    bool started_copy                                                         = false; // if index is found on host copy is not needed
    bool started_copy_from_previous_step                                      = false;

    for (const IndexDescriptor& descriptor_of_index_to_cache : descriptors_of_indices_to_cache)
    {
        const bool host_copy_needed   = !skip_copy_to_host;
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
            index_on_device = Index::create_index(allocator_,
                                                  *parser,
                                                  descriptor_of_index_to_cache.first_read(),
                                                  descriptor_of_index_to_cache.first_read() + descriptor_of_index_to_cache.number_of_reads(),
                                                  kmer_size_,
                                                  window_size_,
                                                  hash_representations_,
                                                  filtering_parameter_,
                                                  cuda_stream_generation_);

            // wait for index to be generated on cuda_stream_generation_ before copying it on cuda_stream_copy_
            // TODO: do this sync using an event
            GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream_generation_));

            if (host_copy_needed)
            {
                // if a D2H copy has been been started in the previous step wait for it to finish
                if (started_copy_from_previous_step)
                {
                    GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream_copy_));
                    // copying is done, these pointer are not needed anymore
                    index_on_host_from_previous_step   = nullptr;
                    index_on_device_from_previous_step = nullptr;
                }

                index_on_host = IndexHostCopy::create_cache(*index_on_device,
                                                            descriptor_of_index_to_cache.first_read(),
                                                            kmer_size_,
                                                            window_size_,
                                                            cuda_stream_copy_);
                started_copy  = true; // index is being copied to host memory, a sync will be needed
            }
        }

        if (host_copy_needed)
        {
            assert(index_on_host);
            new_cache[descriptor_of_index_to_cache] = index_on_host;
        }

        // Device copy of index is only saved if is already exists, i.e. if the index has been generated
        // If the index has been found on host it won't be copied back to device at this point
        // TODO: check device caches from this index in that case, this is not expected to happen frequently so performance gains are going to be small
        if (device_copy_needed && index_on_device)
        {
            indices_kept_on_device[descriptor_of_index_to_cache] = index_on_device;
        }

        // if a D2H copy has been been started in the previous step and it has not been waited for yet wait for it to finish
        if (started_copy_from_previous_step && index_on_host_from_previous_step && index_on_device_from_previous_step)
        {
            GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream_copy_));
            // copying is done, these pointer are not needed anymore
            index_on_host_from_previous_step   = nullptr;
            index_on_device_from_previous_step = nullptr;
        }

        // prepare for next step
        started_copy_from_previous_step = started_copy;
        if (started_copy)
        {
            index_on_host_from_previous_step   = index_on_host;
            index_on_device_from_previous_step = index_on_device;
        }
        else
        {
            index_on_host_from_previous_step   = nullptr;
            index_on_device_from_previous_step = nullptr;
        }
        index_on_host   = nullptr;
        index_on_device = nullptr;
    }

    // wait for the last copy to finish
    if (started_copy_from_previous_step)
    {
        GW_CU_CHECK_ERR(cudaStreamSynchronize(cuda_stream_copy_));
    }

    std::swap(new_cache, this_cache);
}

void IndexCache::start_generating_content_device(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                                 const CacheSelector which_cache)
{
    const host_cache_t& host_cache           = (CacheSelector::query_cache == which_cache) ? query_host_cache_ : target_host_cache_;
    device_cache_t& this_device_cache        = (CacheSelector::query_cache == which_cache) ? query_device_cache_ : target_device_cache_;
    const device_cache_t& other_device_cache = (CacheSelector::query_cache == which_cache) ? target_device_cache_ : query_device_cache_;
    device_cache_t& indices_kept_on_device   = (CacheSelector::query_cache == which_cache) ? query_indices_kept_on_device_ : target_indices_kept_on_device_;
    device_cache_t& new_cache                = (CacheSelector::query_cache == which_cache) ? next_query_device_cache_ : next_target_device_cache_;

    for (const IndexDescriptor& index_descriptor : descriptors_of_indices_to_cache)
    {
        std::shared_ptr<Index> device_index = nullptr;

        // check if index was kept on device after creation
        auto index_kept_on_device = indices_kept_on_device.find(index_descriptor);
        if (index_kept_on_device != indices_kept_on_device.end())
        {
            device_index = index_kept_on_device->second;
            indices_kept_on_device.erase(index_descriptor);
        }

        // check if value is already in this cache
        if (!device_index)
        {
            auto index_in_this_cache = this_device_cache.find(index_descriptor);
            if (index_in_this_cache != this_device_cache.end())
            {
                device_index = index_in_this_cache->second;
            }
        }

        // if query and target files are the same check the other index as well
        if (!device_index && same_query_and_target_)
        {
            auto index_in_other_cache = other_device_cache.find(index_descriptor);
            if (index_in_other_cache != other_device_cache.end())
            {
                device_index = index_in_other_cache->second;
            }
        }

        // if index has not been found on device copy it from host
        if (!device_index)
        {
            // TODO: Throw a custom exception if index not found instead of std::out_of_range
            std::shared_ptr<const IndexHostCopyBase> index_on_host = host_cache.at(index_descriptor);
            device_index                                           = index_on_host->copy_index_to_device(allocator_, cuda_stream_copy_);
        }

        assert(device_index);
        new_cache[index_descriptor] = device_index;
    }
}

void IndexCache::finish_generating_content_device(const CacheSelector which_cache)
{
    device_cache_t& this_device_cache = (CacheSelector::query_cache == which_cache) ? query_device_cache_ : target_device_cache_;
    device_cache_t& new_cache         = (CacheSelector::query_cache == which_cache) ? next_query_device_cache_ : next_target_device_cache_;

    for (const auto device_index : new_cache)
    {
        device_index.second->wait_to_be_ready();
    }

    this_device_cache.clear();
    std::swap(this_device_cache, new_cache);
}

std::shared_ptr<const Index> IndexCache::get_index_from_cache(const IndexDescriptor& index_descriptor,
                                                              const CacheSelector which_cache) const
{
    const device_cache_t& this_device_cache = (CacheSelector::query_cache == which_cache) ? query_device_cache_ : target_device_cache_;
    // TODO: Throw a custom exception if index not found instead of std::out_of_range
    return this_device_cache.at(index_descriptor);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
