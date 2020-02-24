/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <claragenomics/cudamapper/sketch_element.hpp>
#include <claragenomics/cudamapper/types.hpp>
#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/utils/device_buffer.hpp>

namespace claragenomics
{

namespace cudamapper
{
/// \addtogroup cudamapper
/// \{

/// Index - manages mapping of (k,w)-kmer-representation and all its occurences
class Index
{
public:
    /// \brief Virtual destructor
    virtual ~Index() = default;

    /// \brief returns an array of representations of sketch elements
    /// \return an array of representations of sketch elements
    virtual const device_buffer<representation_t>& representations() const = 0;

    /// \brief returns an array of reads ids for sketch elements
    /// \return an array of reads ids for sketch elements
    virtual const device_buffer<read_id_t>& read_ids() const = 0;

    /// \brief returns an array of starting positions of sketch elements in their reads
    /// \return an array of starting positions of sketch elements in their reads
    virtual const device_buffer<position_in_read_t>& positions_in_reads() const = 0;

    /// \brief returns an array of directions in which sketch elements were read
    /// \return an array of directions in which sketch elements were read
    virtual const device_buffer<SketchElement::DirectionOfRepresentation>& directions_of_reads() const = 0;

    /// \brief returns read name of read with the given read_id
    /// \param read_id
    /// \return read name of read with the given read_id
    virtual const std::string& read_id_to_read_name(const read_id_t read_id) const = 0;

    /// \brief returns an array where each representation is recorder only once, sorted by representation
    /// \return an array where each representation is recorder only once, sorted by representation
    virtual const device_buffer<representation_t>& unique_representations() const = 0;

    /// \brief returns first occurrence of corresponding representation from unique_representations() in data arrays
    /// \return first occurrence of corresponding representation from unique_representations() in data arrays
    virtual const device_buffer<std::uint32_t>& first_occurrence_of_representations() const = 0;

    /// \brief returns read length for the read with the gived read_id
    /// \param read_id
    /// \return read length for the read with the gived read_id
    virtual const std::uint32_t& read_id_to_read_length(const read_id_t read_id) const = 0;

    /// \brief returns number of reads in input data
    /// \return number of reads in input data
    virtual read_id_t number_of_reads() const = 0;

    /// \brief returns look up table array mapping read id to read name
    /// \return the array mapping read id to read name
    virtual const std::vector<std::string>& read_ids_to_read_names() const = 0;

    /// \brief returns an array used for mapping read id to the length of the read
    /// \return the array used for mapping read ids to their lengths
    virtual const std::vector<std::uint32_t>& read_ids_to_read_lengths() const = 0;

    /// \brief returns smallest read_id in index
    /// \return smallest read_id in index (0 if empty index)
    virtual read_id_t smallest_read_id() const = 0;

    /// \brief returns largest read_id in index
    /// \return largest read_id in index (0 if empty index)
    virtual read_id_t largest_read_id() const = 0;

    /// \brief returns length of the longest read in this index
    /// \return length of the longest read in this index
    virtual position_in_read_t number_of_basepairs_in_longest_read() const = 0;

    /// \brief Return the maximum kmer length allowable
    /// \return Return the maximum kmer length allowable
    static uint64_t maximum_kmer_size()
    {
        return sizeof(representation_t) * 8 / 2;
    }

    /// \brief generates a mapping of (k,w)-kmer-representation to all of its occurrences for one or more sequences
    /// \param allocator The device memory allocator to use for temporary buffer allocations
    /// \param parser parser for the whole input file (part that goes into this index is determined by first_read_id and past_the_last_read_id)
    /// \param first_read_id read_id of the first read to the included in this index
    /// \param past_the_last_read_id read_id+1 of the last read to be included in this index
    /// \param kmer_size k - the kmer length
    /// \param window_size w - the length of the sliding window used to find sketch elements  (i.e. the number of adjacent kmers in a window, adjacent = shifted by one basepair)
    /// \param hash_representations - if true, hash kmer representations
    /// \param filtering_parameter - filter out all representations for which number_of_sketch_elements_with_that_representation/total_skech_elements >= filtering_parameter, filtering_parameter == 1.0 disables filtering
    /// \return instance of Index
    static std::unique_ptr<Index>
    create_index(std::shared_ptr<DeviceAllocator> allocator,
                 const io::FastaParser& parser,
                 const read_id_t first_read_id,
                 const read_id_t past_the_last_read_id,
                 const std::uint64_t kmer_size,
                 const std::uint64_t window_size,
                 const bool hash_representations  = true,
                 const double filtering_parameter = 1.0);
};

/// IndexHostCopy - Creates and maintains a copy of computed IndexGPU elements on the host, then allows to retrieve target
/// indices from host instead of recomputing them again
///
class IndexHostCopy
{
public:
    /// \brief copy cached index vectors from the host and create an object of Index on GPU
    /// \param allocator asynchronous device allocator used for temporary buffer allocations
    /// \return a pointer to claragenomics::cudamapper::Index
    virtual std::unique_ptr<Index> copy_index_to_device(std::shared_ptr<claragenomics::DeviceAllocator> allocator) = 0;

    /// \brief returns an array of representations of sketch elements (stored on host)
    /// \return an array of representations of sketch elements
    virtual const std::vector<representation_t>& representations() const = 0;

    /// \brief returns an array of reads ids for sketch elements (stored on host)
    /// \return an array of reads ids for sketch elements
    virtual const std::vector<read_id_t>& read_ids() const = 0;

    /// \brief returns an array of starting positions of sketch elements in their reads (stored on host)
    /// \return an array of starting positions of sketch elements in their reads
    virtual const std::vector<position_in_read_t>& positions_in_reads() const = 0;

    /// \brief returns an array of directions in which sketch elements were read (stored on host)
    /// \return an array of directions in which sketch elements were read
    virtual const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads() const = 0;

    /// \brief returns an array where each representation is recorded only once, sorted by representation (stored on host)
    /// \return an array where each representation is recorded only once, sorted by representation
    virtual const std::vector<representation_t>& unique_representations() const = 0;

    /// \brief returns first occurrence of corresponding representation from unique_representations(), plus one more element with the total number of sketch elements (stored on host)
    /// \return first occurrence of corresponding representation from unique_representations(), plus one more element with the total number of sketch elements
    virtual const std::vector<std::uint32_t>& first_occurrence_of_representations() const = 0;

    /// \brief returns look up table array mapping read id to read name
    /// \return the array mapping read id to read name
    virtual const std::vector<std::string>& read_id_to_read_names() const = 0;

    /// \brief returns an array used for mapping read id to the length of the read
    /// \return the array used for mapping read ids to their lengths
    virtual const std::vector<std::uint32_t>& read_id_to_read_lengths() const = 0;

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

    /// \brief Constructor
    /// \brief creates a copy of input processed index on the host
    /// \param index - pointer to computed index parameters (vectors of sketch elements) on GPU
    /// \param first_read_id - representing smallest read_id in index
    /// \param kmer_size - number of basepairs in a k-mer
    /// \param window_size the number of adjacent k-mers in a window, adjacent = shifted by one basepair
    /// \return - an instance of IndexHostCopy
    static std::unique_ptr<IndexHostCopy>
    create_cache(const Index& index,
                 const read_id_t first_read_id,
                 const std::uint64_t kmer_size,
                 const std::uint64_t window_size);
};

} // namespace cudamapper

} // namespace claragenomics
