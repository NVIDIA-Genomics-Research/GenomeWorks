/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <utility>
#include <unordered_map>

#include <claragenomics/cudamapper/types.hpp>
#include <claragenomics/utils/allocator.hpp>

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

/// IndexDescriptor - Every Index is defined by its first read and the number of reads
class IndexDescriptor
{
public:
    /// \brief constructor
    IndexDescriptor(read_id_t first_read, read_id_t number_of_reads);

    IndexDescriptor(const IndexDescriptor&) = default;
    IndexDescriptor& operator=(const IndexDescriptor&) = default;
    IndexDescriptor(IndexDescriptor&&)                 = default;
    IndexDescriptor& operator=(IndexDescriptor&&) = default;
    ~IndexDescriptor()                            = default;

    /// \brief getter
    read_id_t first_read() const;

    /// \brief getter
    read_id_t number_of_reads() const;

    /// \brief returns hash value
    std::size_t get_hash() const;

private:
    /// \brief generates hash
    void generate_hash();

    /// first read in index
    read_id_t first_read_;
    /// number of reads in index
    read_id_t number_of_reads_;
    /// hash of this object
    std::size_t hash_;
};

/// \brief equality operator
bool operator==(const IndexDescriptor& lhs,
                const IndexDescriptor& rhs);

/// \brief inequality operator
bool operator!=(const IndexDescriptor& lhs,
                const IndexDescriptor& rhs);

/// IndexDescriptorHash - operator() calculates hash of a given IndexDescriptor
struct IndexDescriptorHash
{
    /// \brief caclulates hash of given IndexDescriptor
    std::size_t operator()(const IndexDescriptor& index_descriptor) const;
};

/// IndexCacheHost - Creates Indices, stores them in host memory and on demand copies them back to device memory
class IndexCacheHost
{
public:
    /// \brief Constructor
    /// \param reuse_data if true that means that both query and target are the same and if both query and target cache the same index it will be reused
    /// \param allocator allocator to use for device arrays
    /// \param query_parser
    /// \param target_parser
    /// \param k // see Index
    /// \param w // see Index
    /// \param hash_representations // see Index
    /// \param filtering_parameter // see Index
    /// \param cuda_stream // device memory used for Index copy will only we freed up once all previously scheduled work on this stream has finished
    IndexCacheHost(const bool reuse_data,
                   claragenomics::DefaultDeviceAllocator allocator,
                   std::shared_ptr<claragenomics::io::FastaParser> query_parser,
                   std::shared_ptr<claragenomics::io::FastaParser> target_parser,
                   const std::uint64_t k,
                   const std::uint64_t w,
                   const bool hash_representations  = true,
                   const double filtering_parameter = 1.0,
                   const cudaStream_t cuda_stream   = 0);

    IndexCacheHost(const IndexCacheHost&) = delete;
    IndexCacheHost& operator=(const IndexCacheHost&) = delete;
    IndexCacheHost(IndexCacheHost&&)                 = delete;
    IndexCacheHost& operator=(IndexCacheHost&&) = delete;
    ~IndexCacheHost()                           = default;

    /// \brief Discards previously cached query Indices, creates new Indices and copies them to host memory
    void update_query_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache);

    /// \brief Discards previously cached target Indices, creates new Indices and copies them to host memory
    void update_target_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache);

    /// \brief Copies request Index to device memory
    /// throws if that index is currently not in cache
    std::shared_ptr<Index> get_index_for_query_cache(const IndexDescriptor& descriptor_of_index_to_cache);

    /// \brief Copies request Index to device memory
    /// throws if that index is currently not in cache
    std::shared_ptr<Index> get_index_for_target_cache(const IndexDescriptor& descriptor_of_index_to_cache);

private:
    using cache_type_t = std::unordered_map<IndexDescriptor,
                                            std::shared_ptr<const IndexHostCopyBase>,
                                            IndexDescriptorHash>;

    enum class CacheToUpdate
    {
        QUERY,
        TARGET
    };

    /// \brief Discards previously cached Indices, creates new Indices and copies them to host memory
    /// Uses which_cache to determine if it should be working on query of target indices
    ///
    /// If reuse_data_ is true function checks the other cache to see if that index is already in cache
    void update_cache(const std::vector<IndexDescriptor>& descriptors_of_indices_to_cache,
                      const CacheToUpdate which_cache);

    cache_type_t query_cache_;
    cache_type_t target_cache_;

    const bool reuse_data_;
    claragenomics::DefaultDeviceAllocator allocator_;
    std::shared_ptr<claragenomics::io::FastaParser> query_parser_;
    std::shared_ptr<claragenomics::io::FastaParser> target_parser_;
    const std::uint64_t k_;
    const std::uint64_t w_;
    const bool hash_representations_;
    const double filtering_parameter_;
    const cudaStream_t cuda_stream_;
};

} // namespace cudamapper
} // namespace claragenomics