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

#include "index_host_copy.cu"

#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/io/fasta_parser.hpp>

namespace claragenomics
{
namespace cudamapper
{

IndexDescriptor::IndexDescriptor(read_id_t first_read, read_id_t number_of_reads)
    : first_read_(first_read)
    , number_of_reads_(number_of_reads)
{
    generate_hash();
}

read_id_t IndexDescriptor::first_read() const
{
    return first_read_;
}

read_id_t IndexDescriptor::number_of_reads() const
{
    return number_of_reads_;
}

std::size_t IndexDescriptor::get_hash() const
{
    return hash_;
}

void IndexDescriptor::generate_hash()
{
    // populate lower half of hash with one value, upper half with the other
    std::size_t bytes_per_element = sizeof(std::size_t) / 2;
    // first half of element_mask are zeros, second are ones
    std::size_t element_mask = (1 << (8 * bytes_per_element)) - 1;
    // set lower bits
    hash_ |= first_read_ & element_mask;
    // set higher bits
    hash_ |= (number_of_reads_ & element_mask) << (8 * bytes_per_element);
}

bool operator==(const IndexDescriptor& lhs, const IndexDescriptor& rhs)
{
    return lhs.first_read() == rhs.first_read() && lhs.number_of_reads() == rhs.number_of_reads();
}

bool operator!=(const IndexDescriptor& lhs, const IndexDescriptor& rhs)
{
    return !(lhs == rhs);
}

std::size_t IndexDescriptorHash::operator()(const IndexDescriptor& index_descriptor) const
{
    return index_descriptor.get_hash();
}

IndexCacheHost::IndexCacheHost(const bool reuse_data,
                               claragenomics::DefaultDeviceAllocator allocator,
                               std::shared_ptr<claragenomics::io::FastaParser> query_parser,
                               std::shared_ptr<claragenomics::io::FastaParser> target_parser,
                               const std::uint64_t k,
                               const std::uint64_t w,
                               const bool hash_representations,
                               const double filtering_parameter,
                               const cudaStream_t cuda_stream)
    : reuse_data_(reuse_data)
    , allocator_(allocator)
    , query_parser_(query_parser)
    , target_parser_(target_parser)
    , k_(k)
    , w_(w)
    , hash_representations_(hash_representations)
    , filtering_parameter_(filtering_parameter)
    , cuda_stream_(cuda_stream)
{
}

void IndexCacheHost::update_query_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    update_cache(descriptors_of_indices_to_cache, CacheToUpdate::QUERY);
}

void IndexCacheHost::update_target_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache)
{
    update_cache(descriptors_of_indices_to_cache, CacheToUpdate::TARGET);
}

std::shared_ptr<Index> IndexCacheHost::get_index_for_query_cache(const IndexDescriptor& descriptor_of_index_to_cache)
{
    return query_cache_.at(descriptor_of_index_to_cache)->copy_index_to_device(allocator_, cuda_stream_);
}

std::shared_ptr<Index> IndexCacheHost::get_index_for_target_cache(const IndexDescriptor& descriptor_of_index_to_cache)
{
    return target_cache_.at(descriptor_of_index_to_cache)->copy_index_to_device(allocator_, cuda_stream_);
}

void IndexCacheHost::update_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                                  const CacheToUpdate which_cache)
{
    cache_type_t& cache_to_edit                            = (CacheToUpdate::QUERY == which_cache) ? query_cache_ : target_cache_;
    const cache_type_t& cache_to_check                     = (CacheToUpdate::QUERY == which_cache) ? target_cache_ : query_cache_;
    std::shared_ptr<claragenomics::io::FastaParser> parser = (CacheToUpdate::QUERY == which_cache) ? query_parser_ : target_parser_;

    cache_to_edit.clear();

    for (const IndexDescriptor& descriptor_of_index_to_cache : descriptors_of_indices_to_cache)
    {

        std::shared_ptr<const IndexHostCopyBase> index_copy = nullptr;

        if (reuse_data_)
        {
            // check if the same index already exists on host
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
                                                                        k_,
                                                                        w_,
                                                                        hash_representations_,
                                                                        filtering_parameter_,
                                                                        cuda_stream_);
            // copy it to host memory
            index_copy = claragenomics::cudamapper::IndexHostCopy::create_cache(*index,
                                                                                descriptor_of_index_to_cache.first_read(),
                                                                                k_,
                                                                                w_,
                                                                                cuda_stream_);
        }

        assert(nullptr != index_copy);

        // save pointer to cached index
        cache_to_edit[descriptor_of_index_to_cache] = index_copy;
    }
}

} // namespace cudamapper
} // namespace claragenomics