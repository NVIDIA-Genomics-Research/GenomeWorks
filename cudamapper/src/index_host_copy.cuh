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

#include <claragenomics/cudamapper/index.hpp>

namespace claragenomics
{
namespace cudamapper
{
/// IndexHostCopy - Creates and maintains a copy of computed IndexGPU elements on the host
///
///
class IndexHostCopy : public IndexHostCopyBase
{
public:
    /// \brief Constructor
    /// \brief cache the computed index to host
    /// \param index - pointer to computed index parameters (vectors of sketch elements) on GPU
    /// \param first_read_id - representing smallest read_id in index
    /// \param kmer_size - number of basepairs in a k-mer
    /// \param window_size the number of adjacent k-mers in a window, adjacent = shifted by one basepair
    /// \param cuda_stream D2H copy is done on this stream
    /// \return - pointer to claragenomics::cudamapper::IndexCache
    IndexHostCopy(const Index& index,
                  const read_id_t first_read_id,
                  const std::uint64_t kmer_size,
                  const std::uint64_t window_size,
                  const cudaStream_t cuda_stream);

    /// \brief copy cached index vectors from the host and create an object of Index on GPU
    /// \param allocator pointer to asynchronous device allocator
    /// \param cuda_stream H2D copy is done on this stream. Device arrays are also associated with this stream and will not be freed at least until all work issued on this stream before calling their destructor is done
    /// \return a pointer to claragenomics::cudamapper::Index
    std::unique_ptr<Index> copy_index_to_device(DefaultDeviceAllocator allocator,
                                                const cudaStream_t cuda_stream = 0) const override;

    /// \brief returns an array of representations of sketch elements (stored on host)
    /// \return an array of representations of sketch elements
    const std::vector<representation_t>& representations() const override;

    /// \brief returns an array of reads ids for sketch elements (stored on host)
    /// \return an array of reads ids for sketch elements
    const std::vector<read_id_t>& read_ids() const override;

    /// \brief returns an array of starting positions of sketch elements in their reads (stored on host)
    /// \return an array of starting positions of sketch elements in their reads
    const std::vector<position_in_read_t>& positions_in_reads() const override;

    /// \brief returns an array of directions in which sketch elements were read (stored on host)
    /// \return an array of directions in which sketch elements were read
    const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads() const override;

    /// \brief returns an array where each representation is recorded only once, sorted by representation (stored on host)
    /// \return an array where each representation is recorded only once, sorted by representation
    const std::vector<representation_t>& unique_representations() const override;

    /// \brief returns first occurrence of corresponding representation from unique_representations(), plus one more element with the total number of sketch elements (stored on host)
    /// \return first occurrence of corresponding representation from unique_representations(), plus one more element with the total number of sketch elements
    const std::vector<std::uint32_t>& first_occurrence_of_representations() const override;

    /// \brief returns number of reads in input data
    /// \return number of reads in input data
    read_id_t number_of_reads() const override;

    /// \brief returns length of the longest read in this index
    /// \return length of the longest read in this index
    position_in_read_t number_of_basepairs_in_longest_read() const override;

    /// \brief returns stored value in first_read_id_ representing smallest read_id in index
    /// \return first_read_id_
    read_id_t first_read_id() const override;

    /// \brief returns k-mer size
    /// \return kmer_size_
    std::uint64_t kmer_size() const override;

    /// \brief returns window size
    /// \return window_size_
    std::uint64_t window_size() const override;

private:
    std::vector<representation_t> representations_;
    std::vector<read_id_t> read_ids_;
    std::vector<position_in_read_t> positions_in_reads_;
    std::vector<SketchElement::DirectionOfRepresentation> directions_of_reads_;

    std::vector<representation_t> unique_representations_;
    std::vector<std::uint32_t> first_occurrence_of_representations_;

    read_id_t number_of_reads_;
    position_in_read_t number_of_basepairs_in_longest_read_;

    const read_id_t first_read_id_   = 0;
    const std::uint64_t kmer_size_   = 0;
    const std::uint64_t window_size_ = 0;
};

} // namespace cudamapper
} // namespace claragenomics