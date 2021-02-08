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

#pragma once

#include <exception>
#include <memory>
#include <string>
#include <claraparabricks/genomeworks/cudamapper/sketch_element.hpp>
#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include <claraparabricks/genomeworks/io/fasta_parser.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
/// \addtogroup cudamapper
/// \{

/// IndexDescriptor Implementation - Every Index is defined by its first read and the number of reads
/// Used in conjunction with a fasta parser object to create an index
class IndexDescriptor
{
public:
    /// \brief constructor
    IndexDescriptor(read_id_t first_read,
                    number_of_reads_t number_of_reads);

    /// \brief getter
    read_id_t first_read() const;

    /// \brief getter
    number_of_reads_t number_of_reads() const;

    /// \brief returns hash value
    std::size_t get_hash() const;

private:
    /// \brief generates hash
    void generate_hash();

    /// first read in index
    read_id_t first_read_;
    /// number of reads in index
    number_of_reads_t number_of_reads_;
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

/// Index - manages mapping of (k,w)-kmer-representation and all its occurences
class Index
{
public:
    /// \brief Virtual destructor
    virtual ~Index() = default;

    /// \brief returns an array of representations of sketch elements
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return an array of representations of sketch elements
    virtual const device_buffer<representation_t>& representations() const = 0;

    /// \brief returns an array of reads ids for sketch elements
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return an array of reads ids for sketch elements
    virtual const device_buffer<read_id_t>& read_ids() const = 0;

    /// \brief returns an array of starting positions of sketch elements in their reads
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return an array of starting positions of sketch elements in their reads
    virtual const device_buffer<position_in_read_t>& positions_in_reads() const = 0;

    /// \brief returns an array of directions in which sketch elements were read
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return an array of directions in which sketch elements were read
    virtual const device_buffer<SketchElement::DirectionOfRepresentation>& directions_of_reads() const = 0;

    /// \brief returns an array where each representation is recorder only once, sorted by representation
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return an array where each representation is recorder only once, sorted by representation
    virtual const device_buffer<representation_t>& unique_representations() const = 0;

    /// \brief returns first occurrence of corresponding representation from unique_representations() in data arrays
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return first occurrence of corresponding representation from unique_representations() in data arrays
    virtual const device_buffer<std::uint32_t>& first_occurrence_of_representations() const = 0;

    /// \brief returns number of reads in input data
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return number of reads in input data
    virtual read_id_t number_of_reads() const = 0;

    /// \brief returns smallest read_id in index
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return smallest read_id in index (0 if empty index)
    virtual read_id_t smallest_read_id() const = 0;

    /// \brief returns largest read_id in index
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return largest read_id in index (0 if empty index)
    virtual read_id_t largest_read_id() const = 0;

    /// \brief returns length of the longest read in this index
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return length of the longest read in this index
    virtual position_in_read_t number_of_basepairs_in_longest_read() const = 0;

    /// \brief Return the maximum kmer length allowable
    /// \throw IndexNotReadyException if called before wait_to_be_ready()
    /// \return Return the maximum kmer length allowable
    static uint64_t maximum_kmer_size()
    {
        return sizeof(representation_t) * CHAR_BIT / 2;
    }

    /// \brief checks if index is ready to be used on this device, index might not be ready if its creation or copy from host is asynchronous
    /// \return true if the index is ready to be used
    virtual bool is_ready() const = 0;

    /// \brief if is_ready() is true returns immediately, blocks until it becomes ready otherwise
    virtual void wait_to_be_ready() = 0;

    /// \brief generates a mapping of (k,w)-kmer-representation to all of its occurrences for one or more sequences
    /// Generation is done asynchronously and one should call wait_to_be_ready() before using the object
    /// \param allocator The device memory allocator to use for temporary buffer allocations
    /// \param parser parser for the whole input file (part that goes into this index is determined by first_read_id and past_the_last_read_id)
    /// \param descriptor object holding the first read, size of the read, and hash value of the index being created
    /// \param kmer_size k - the kmer length
    /// \param window_size w - the length of the sliding window used to find sketch elements  (i.e. the number of adjacent kmers in a window, adjacent = shifted by one basepair)
    /// \param hash_representations if true, hash kmer representations
    /// \param filtering_parameter filter out all representations for which number_of_sketch_elements_with_that_representation/total_skech_elements >= filtering_parameter, filtering_parameter == 1.0 disables filtering
    /// \param cuda_stream_generation CUDA stream on which index is generated. Device arrays are associated with this stream and will not be freed at least until all work issued on this stream before calling their destructor has been done
    /// \param cuda_stream_copy CUDA stream on which the index is copied to host if needed. Device arrays are associated with this stream and will not be freed at least until all work issued on this stream before calling their destructor has been done
    /// \return instance of Index
    static std::unique_ptr<Index>
    create_index_async(DefaultDeviceAllocator allocator,
                       const io::FastaParser& parser,
                       const IndexDescriptor& descriptor,
                       const std::uint64_t kmer_size,
                       const std::uint64_t window_size,
                       const bool hash_representations           = true,
                       const double filtering_parameter          = 1.0,
                       const cudaStream_t cuda_stream_generation = 0,
                       const cudaStream_t cuda_stream_copy       = 0);
};

/// IndexHostCopyBase - Creates and maintains a copy of computed IndexGPU elements on the host, then allows to retrieve target
/// indices from host instead of recomputing them again
///
class IndexHostCopyBase
{
public:
    /// Span - helper struct that provides a view of part of underlying array
    /// \tparam T type of data in this view
    template <typename T>
    struct Span
    {
        /// pointer to first element in view
        T* data;
        /// total elements of type T in view
        std::ptrdiff_t size;
    };

    /// \brief copy cached index vectors from the host and create an object of Index on GPU
    /// \param allocator asynchronous device allocator used for temporary buffer allocations
    /// \param cuda_stream H2D copy is done on this stream. Device arrays are also associated with this stream and will not be freed at least until all work issued on this stream before calling their destructor is done
    /// \return a pointer to genomeworks::cudamapper::Index
    virtual std::unique_ptr<Index> copy_index_to_device(DefaultDeviceAllocator allocator,
                                                        const cudaStream_t cuda_stream = 0) const = 0;

    /// \brief virtual destructor
    virtual ~IndexHostCopyBase() = default;

    /// \brief waits for copy to host to be done
    virtual void finish_copying() const = 0;

    /// \brief returns an array of representations of sketch elements (stored on host)
    /// \return an array of representations of sketch elements
    virtual const Span<representation_t> representations() const = 0;

    /// \brief returns an array of reads ids for sketch elements (stored on host)
    /// \return an array of reads ids for sketch elements
    virtual const Span<read_id_t> read_ids() const = 0;

    /// \brief returns an array of starting positions of sketch elements in their reads (stored on host)
    /// \return an array of starting positions of sketch elements in their reads
    virtual const Span<position_in_read_t> positions_in_reads() const = 0;

    /// \brief returns an array of directions in which sketch elements were read (stored on host)
    /// \return an array of directions in which sketch elements were read
    virtual const Span<SketchElement::DirectionOfRepresentation> directions_of_reads() const = 0;

    /// \brief returns an array where each representation is recorded only once, sorted by representation (stored on host)
    /// \return an array where each representation is recorded only once, sorted by representation
    virtual const Span<representation_t> unique_representations() const = 0;

    /// \brief returns first occurrence of corresponding representation from unique_representations(), plus one more element with the total number of sketch elements (stored on host)
    /// \return first occurrence of corresponding representation from unique_representations(), plus one more element with the total number of sketch elements
    virtual const Span<std::uint32_t> first_occurrence_of_representations() const = 0;

    /// \brief returns number of reads in input data
    /// \return number of reads in input data
    virtual read_id_t number_of_reads() const = 0;

    /// \brief returns length of the longest read in this index
    /// \return length of the longest read in this index
    virtual position_in_read_t number_of_basepairs_in_longest_read() const = 0;

    /// \brief returns stored value in first_read_id_ representing smallest read_id in index
    /// \return first_read_id_
    virtual read_id_t first_read_id() const = 0;

    /// \brief returns k-mer size
    /// \return kmer_size_
    virtual std::uint64_t kmer_size() const = 0;

    /// \brief returns window size
    /// \return window_size_
    virtual std::uint64_t window_size() const = 0;

    /// \brief Starts creating a copy of index on the host
    /// Copy is done asynchronously and one should wait for it to finish with finish_copying()
    /// \param index - pointer to computed index parameters (vectors of sketch elements) on GPU
    /// \param first_read_id - representing smallest read_id in index
    /// \param kmer_size - number of basepairs in a k-mer
    /// \param window_size the number of adjacent k-mers in a window, adjacent = shifted by one basepair
    /// \param cuda_stream D2H copy is done on this stream
    /// \return - an instance of IndexHostCopyBase
    static std::unique_ptr<IndexHostCopyBase> create_host_copy_async(const Index& index,
                                                                     const read_id_t first_read_id,
                                                                     const std::uint64_t kmer_size,
                                                                     const std::uint64_t window_size,
                                                                     const cudaStream_t cuda_stream = 0);
};

/// IndexNotReadyException - Exception to be thrown if a member of Index is accessed, but is_read() == false
class IndexNotReadyException : public std::exception
{
public:
    /// \brief Constructor
    /// \param function_name name of funciton which has been accessed
    IndexNotReadyException(const std::string& function_name);

    /// \brief Returns the error message of the exception
    const char* what() const noexcept override;

private:
    const std::string message_;
};

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
