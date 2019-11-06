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

#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "claragenomics/cudamapper/index_two_indices.hpp"
#include "claragenomics/cudamapper/types.hpp"
#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/logging/logging.hpp>
#include <claragenomics/utils/device_buffer.cuh>
#include <claragenomics/utils/mathutils.hpp>

namespace claragenomics
{
namespace cudamapper
{
/// IndexGPU - Contains sketch elements grouped by representation and by read id within the representation
///
/// Class contains four separate data arrays: representations, read_ids, positions_in_reads and directions_of_reads.
/// Elements of these four arrays with the same index represent one sketch element
/// (representation, read_id of the read it belongs to, position in that read of the first basepair of sketch element and whether it is forward or reverse complement representation).
///
/// Elements of data arrays are grouped by sketch element representation and within those groups by read_id. Both representations and read_ids within representations are sorted in ascending order
///
/// \tparam SketchElementImpl any implementation of SketchElement
template <typename SketchElementImpl>
class IndexGPUTwoIndices : public IndexTwoIndices
{
public:
    /// \brief Constructor
    ///
    /// \param parser parser for the whole input file (part that goes into this index is determined by first_read_id and past_the_last_read_id)
    /// \param first_read_id read_id of the first read to the included in this index
    /// \param past_the_last_read_id read_id+1 of the last read to be included in this index
    /// \param kmer_size k - the kmer length
    /// \param window_size w - the length of the sliding window used to find sketch elements
    IndexGPUTwoIndices(io::FastaParser* parser,
                       const read_id_t first_read_id,
                       const read_id_t past_the_last_read_id,
                       const std::uint64_t kmer_size,
                       const std::uint64_t window_size);

    /// \brief Constructor
    IndexGPUTwoIndices();

    /// \brief returns an array of representations of sketch elements
    /// \return an array of representations of sketch elements
    const thrust::device_vector<representation_t>& representations() const override;

    /// \brief returns an array of reads ids for sketch elements
    /// \return an array of reads ids for sketch elements
    const thrust::device_vector<read_id_t>& read_ids() const override;

    /// \brief returns an array of starting positions of sketch elements in their reads
    /// \return an array of starting positions of sketch elements in their reads
    const thrust::device_vector<position_in_read_t>& positions_in_reads() const override;

    /// \brief returns an array of directions in which sketch elements were read
    /// \return an array of directions in which sketch elements were read
    const thrust::device_vector<typename SketchElementImpl::DirectionOfRepresentation>& directions_of_reads() const override;

    /// \brief returns read name of read with the given read_id
    /// \param read_id
    /// \return read name of read with the given read_id
    const std::string& read_id_to_read_name(const read_id_t read_id) const override;

    /// \brief returns read length for the read with the gived read_id
    /// \param read_id
    /// \return read length for the read with the gived read_id
    const std::uint32_t& read_id_to_read_length(const read_id_t read_id) const override;

    /// \brief returns number of reads in input data
    /// \return number of reads in input data
    std::uint64_t number_of_reads() const override;

private:
    /// \brief generates the index
    void generate_index(io::FastaParser* query_parser,
                        const read_id_t first_read_id,
                        const read_id_t past_the_last_read_id);

    thrust::device_vector<representation_t> representations_d_;
    thrust::device_vector<read_id_t> read_ids_d_;
    thrust::device_vector<position_in_read_t> positions_in_reads_d_;
    thrust::device_vector<typename SketchElementImpl::DirectionOfRepresentation> directions_of_reads_d_;

    std::vector<std::string> read_id_to_read_name_;
    std::vector<std::uint32_t> read_id_to_read_length_;

    const read_id_t first_read_id_;
    const std::uint64_t kmer_size_;
    const std::uint64_t window_size_;
    std::uint64_t number_of_reads_;
};

namespace details
{
namespace index_gpu_two_indices
{

/// \brief Splits array of structs into one array per struct element
///
/// \param rest_d original struct
/// \param positions_in_reads_d output array
/// \param read_ids_d output array
/// \param directions_of_reads_d output array
/// \param total_elements number of elements in each array
///
/// \tparam ReadidPositionDirection any implementation of SketchElementImpl::ReadidPositionDirection
/// \tparam DirectionOfRepresentation any implementation of SketchElementImpl::SketchElementImpl::DirectionOfRepresentation
template <typename ReadidPositionDirection, typename DirectionOfRepresentation>
__global__ void copy_rest_to_separate_arrays(const ReadidPositionDirection* const rest_d,
                                             read_id_t* const read_ids_d,
                                             position_in_read_t* const positions_in_reads_d,
                                             DirectionOfRepresentation* const directions_of_reads_d,
                                             const std::size_t total_elements)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= total_elements)
        return;

    read_ids_d[i]            = rest_d[i].read_id_;
    positions_in_reads_d[i]  = rest_d[i].position_in_read_;
    directions_of_reads_d[i] = DirectionOfRepresentation(rest_d[i].direction_);
}

} // namespace index_gpu_two_indices
} // namespace details

template <typename SketchElementImpl>
IndexGPUTwoIndices<SketchElementImpl>::IndexGPUTwoIndices(io::FastaParser* parser,
                                                          const read_id_t first_read_id,
                                                          const read_id_t past_the_last_read_id,
                                                          const std::uint64_t kmer_size,
                                                          const std::uint64_t window_size)
    : first_read_id_(first_read_id)
    , kmer_size_(kmer_size)
    , window_size_(window_size)
    , number_of_reads_(0)
{
    generate_index(parser,
                   first_read_id_,
                   past_the_last_read_id);
}

template <typename SketchElementImpl>
IndexGPUTwoIndices<SketchElementImpl>::IndexGPUTwoIndices()
    : first_read_id_(0)
    , kmer_size_(0)
    , window_size_(0)
    , number_of_reads_(0)
{
}

template <typename SketchElementImpl>
const thrust::device_vector<representation_t>& IndexGPUTwoIndices<SketchElementImpl>::representations() const
{
    return representations_d_;
};

template <typename SketchElementImpl>
const thrust::device_vector<read_id_t>& IndexGPUTwoIndices<SketchElementImpl>::read_ids() const
{
    return read_ids_d_;
}

template <typename SketchElementImpl>
const thrust::device_vector<position_in_read_t>& IndexGPUTwoIndices<SketchElementImpl>::positions_in_reads() const
{
    return positions_in_reads_d_;
}

template <typename SketchElementImpl>
const thrust::device_vector<typename SketchElementImpl::DirectionOfRepresentation>& IndexGPUTwoIndices<SketchElementImpl>::directions_of_reads() const
{
    return directions_of_reads_d_;
}

template <typename SketchElementImpl>
const std::string& IndexGPUTwoIndices<SketchElementImpl>::read_id_to_read_name(const read_id_t read_id) const
{
    return read_id_to_read_name_[read_id - first_read_id_];
}

template <typename SketchElementImpl>
const std::uint32_t& IndexGPUTwoIndices<SketchElementImpl>::read_id_to_read_length(const read_id_t read_id) const
{
    return read_id_to_read_length_[read_id - first_read_id_];
}

template <typename SketchElementImpl>
std::uint64_t IndexGPUTwoIndices<SketchElementImpl>::number_of_reads() const
{
    return number_of_reads_;
}

template <typename SketchElementImpl>
void IndexGPUTwoIndices<SketchElementImpl>::generate_index(io::FastaParser* parser,
                                                           const read_id_t first_read_id,
                                                           const read_id_t past_the_last_read_id)
{

    // check if there are any reads to process
    if (first_read_id >= past_the_last_read_id)
    {
        CGA_LOG_INFO("No Sketch Elements to be added to index");
        number_of_reads_ = 0;
        return;
    }

    number_of_reads_ = past_the_last_read_id - first_read_id;

    std::uint64_t total_basepairs = 0;
    std::vector<ArrayBlock> read_id_to_basepairs_section_h;
    std::vector<io::FastaSequence> fasta_reads;

    // deterine the number of basepairs in each read and assign read_id to each read
    for (read_id_t read_id = first_read_id; read_id < past_the_last_read_id; ++read_id)
    {
        fasta_reads.emplace_back(parser->get_sequence_by_id(read_id));
        const std::string& read_basepairs = fasta_reads.back().seq;
        const std::string& read_name      = fasta_reads.back().name;
        if (read_basepairs.length() >= window_size_ + kmer_size_ - 1)
        {
            read_id_to_basepairs_section_h.emplace_back(ArrayBlock{total_basepairs, static_cast<std::uint32_t>(read_basepairs.length())});
            total_basepairs += read_basepairs.length();
            read_id_to_read_name_.push_back(read_name);
            read_id_to_read_length_.push_back(read_basepairs.length());
        }
        else
        {
            // TODO: Implement this skipping in a correct manner
            CGA_LOG_INFO("Skipping read {}. It has {} basepairs, one window covers {} basepairs",
                         read_name,
                         read_basepairs.length(),
                         window_size_ + kmer_size_ - 1);
        }
    }

    if (0 == total_basepairs)
    {
        CGA_LOG_INFO("Index for reads {} to past {} is empty",
                     first_read_id,
                     past_the_last_read_id);
        number_of_reads_ = 0;
        return;
    }

    std::vector<char> merged_basepairs_h(total_basepairs);

    // copy basepairs from each read into one big array
    // read_id starts from first_read_id which can have an arbitrary value, local_read_id always starts from 0
    for (read_id_t local_read_id = 0; local_read_id < number_of_reads_; ++local_read_id)
    {
        const std::string& read_basepairs = fasta_reads[local_read_id].seq;
        std::copy(std::begin(read_basepairs),
                  std::end(read_basepairs),
                  std::next(std::begin(merged_basepairs_h), read_id_to_basepairs_section_h[local_read_id].first_element_));
    }
    fasta_reads.clear();
    fasta_reads.shrink_to_fit();

    // move basepairs to the device
    CGA_LOG_INFO("Allocating {} bytes for read_id_to_basepairs_section_d", read_id_to_basepairs_section_h.size() * sizeof(decltype(read_id_to_basepairs_section_h)::value_type));
    device_buffer<decltype(read_id_to_basepairs_section_h)::value_type> read_id_to_basepairs_section_d(read_id_to_basepairs_section_h.size());
    CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_basepairs_section_d.data(),
                                read_id_to_basepairs_section_h.data(),
                                read_id_to_basepairs_section_h.size() * sizeof(decltype(read_id_to_basepairs_section_h)::value_type),
                                cudaMemcpyHostToDevice));

    CGA_LOG_INFO("Allocating {} bytes for merged_basepairs_d", merged_basepairs_h.size() * sizeof(decltype(merged_basepairs_h)::value_type));
    device_buffer<decltype(merged_basepairs_h)::value_type> merged_basepairs_d(merged_basepairs_h.size());
    CGA_CU_CHECK_ERR(cudaMemcpy(merged_basepairs_d.data(),
                                merged_basepairs_h.data(),
                                merged_basepairs_h.size() * sizeof(decltype(merged_basepairs_h)::value_type),
                                cudaMemcpyHostToDevice));
    merged_basepairs_h.clear();
    merged_basepairs_h.shrink_to_fit();

    // sketch elements get generated here
    auto sketch_elements                                                      = SketchElementImpl::generate_sketch_elements(number_of_reads_,
                                                                       kmer_size_,
                                                                       window_size_,
                                                                       first_read_id,
                                                                       merged_basepairs_d,
                                                                       read_id_to_basepairs_section_h,
                                                                       read_id_to_basepairs_section_d);
    device_buffer<representation_t> representations_d                         = std::move(sketch_elements.representations_d);
    device_buffer<typename SketchElementImpl::ReadidPositionDirection> rest_d = std::move(sketch_elements.rest_d);

    CGA_LOG_INFO("Deallocating {} bytes from read_id_to_basepairs_section_d", read_id_to_basepairs_section_d.size() * sizeof(decltype(read_id_to_basepairs_section_d)::value_type));
    read_id_to_basepairs_section_d.free();
    CGA_LOG_INFO("Deallocating {} bytes from merged_basepairs_d", merged_basepairs_d.size() * sizeof(decltype(merged_basepairs_d)::value_type));
    merged_basepairs_d.free();

    // *** sort sketch elements by representation ***
    // As this is a stable sort and the data was initailly grouper by read_id this means that the sketch elements within each representations are sorted by read_id
    thrust::stable_sort_by_key(thrust::device,
                               representations_d.data(),
                               representations_d.data() + representations_d.size(),
                               rest_d.data());

    // copy the data to member functions (depending on the interface desing these copies might not be needed)
    representations_d_.resize(representations_d.size());
    representations_d_.shrink_to_fit();
    thrust::copy(thrust::device,
                 representations_d.data(),
                 representations_d.data() + representations_d.size(),
                 representations_d_.begin());
    representations_d.free();

    read_ids_d_.resize(representations_d_.size());
    read_ids_d_.shrink_to_fit();
    positions_in_reads_d_.resize(representations_d_.size());
    positions_in_reads_d_.shrink_to_fit();
    directions_of_reads_d_.resize(representations_d_.size());
    directions_of_reads_d_.shrink_to_fit();

    const std::uint32_t threads = 256;
    const std::uint32_t blocks  = ceiling_divide<int64_t>(representations_d_.size(), threads);

    details::index_gpu_two_indices::copy_rest_to_separate_arrays<<<blocks, threads>>>(rest_d.data(),
                                                                                      read_ids_d_.data().get(),
                                                                                      positions_in_reads_d_.data().get(),
                                                                                      directions_of_reads_d_.data().get(),
                                                                                      representations_d_.size());
    CGA_CU_CHECK_ERR(cudaDeviceSynchronize());
}

} // namespace cudamapper
} // namespace claragenomics
