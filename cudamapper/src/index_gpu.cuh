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

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/replace.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/cudamapper/types.hpp>
#include <claragenomics/io/fasta_parser.hpp>
#include <claragenomics/logging/logging.hpp>
#include <claragenomics/utils/device_buffer.cuh>
#include <claragenomics/utils/mathutils.hpp>
#include <claragenomics/utils/signed_integer_utils.hpp>

namespace claragenomics
{
namespace cudamapper
{
/// IndexGPU - Contains sketch elements grouped by representation and by read id within the representation
///
/// Sketch elements are separated in four data arrays: representations, read_ids, positions_in_reads and directions_of_reads.
/// Elements of these four arrays with the same index represent one sketch element
/// (representation, read_id of the read it belongs to, position in that read of the first basepair of sketch element and whether it is
/// forward or reverse complement representation).
///
/// Elements of data arrays are grouped by sketch element representation and within those groups by read_id. Both representations and read_ids within
/// representations are sorted in ascending order
///
/// In addition to this the class contains an array where each representation is recorder only once (unique_representations) sorted by representation
/// and an array in which the index of first occurrence of that representation is recorded
///
/// \tparam SketchElementImpl any implementation of SketchElement
template <typename SketchElementImpl>
class IndexGPU : public Index
{
public:
    /// \brief Constructor
    ///
    /// \param parser parser for the whole input file (part that goes into this index is determined by first_read_id and past_the_last_read_id)
    /// \param first_read_id read_id of the first read to the included in this index
    /// \param past_the_last_read_id read_id+1 of the last read to be included in this index
    /// \param kmer_size k - the kmer length
    /// \param window_size w - the length of the sliding window used to find sketch elements (i.e. the number of adjacent k-mers in a window, adjacent = shifted by one basepair)
    /// \param hash_representations - if true, hash kmer representations
    /// \param filtering_parameter - filter out all representations for which number_of_sketch_elements_with_that_representation/total_skech_elements >= filtering_parameter, filtering_parameter == 1.0 disables filtering
    IndexGPU(const io::FastaParser& parser,
             const read_id_t first_read_id,
             const read_id_t past_the_last_read_id,
             const std::uint64_t kmer_size,
             const std::uint64_t window_size,
             const bool hash_representations  = true,
             const double filtering_parameter = 1.0);

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

    /// \brief returns an array where each representation is recorder only once, sorted by representation
    /// \return an array where each representation is recorder only once, sorted by representation
    const thrust::device_vector<representation_t>& unique_representations() const override;

    /// \brief returns first occurrence of corresponding representation from unique_representations() in data arrays, plus one more element with the total number of sketch elements
    /// \return first occurrence of corresponding representation from unique_representations() in data arrays, plus one more element with the total number of sketch elements
    const thrust::device_vector<std::uint32_t>& first_occurrence_of_representations() const override;

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
    void generate_index(const io::FastaParser& query_parser,
                        const read_id_t first_read_id,
                        const read_id_t past_the_last_read_id,
                        const bool hash_representations,
                        const double filtering_parameter);

    thrust::device_vector<representation_t> representations_d_;
    thrust::device_vector<read_id_t> read_ids_d_;
    thrust::device_vector<position_in_read_t> positions_in_reads_d_;
    thrust::device_vector<typename SketchElementImpl::DirectionOfRepresentation> directions_of_reads_d_;

    thrust::device_vector<representation_t> unique_representations_d_;
    thrust::device_vector<std::uint32_t> first_occurrence_of_representations_d_;

    std::vector<std::string> read_id_to_read_name_;
    std::vector<std::uint32_t> read_id_to_read_length_;

    const read_id_t first_read_id_ = 0;
    // number of basepairs in a k-mer
    const std::uint64_t kmer_size_ = 0;
    // the number of adjacent k-mers in a window, adjacent = shifted by one basepair
    const std::uint64_t window_size_ = 0;
    std::uint64_t number_of_reads_   = 0;
};

namespace details
{
namespace index_gpu
{
/// \brief Creates compressed representation of index
///
/// Creates two arrays: first one contains a list of unique representations and the second one the index
/// at which that representation occurrs for the first time in the original data.
/// Second element contains one additional elemet at the end, containing the total number of elemets in the original array.
///
/// For example:
/// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
/// 0  0  0  0 12 12 12 12 12 12 23 23 23 32 32 32 32 32 46 46 46   <- input_representations_d
/// ^           ^                 ^        ^              ^       ^
/// gives:
/// 0 12 23 32 46    <- unique_representations_d
/// 0  4 10 13 18 21 <- first_occurrence_index_d
///
/// \param unique_representations_d empty on input, contains one value of each representation on the output
/// \param first_occurrence_index_d empty on input, index of first occurrence of each representation and additional elemnt on the output
/// \param input_representations_d an array of representaton where representations with the same value stand next to each other
void find_first_occurrences_of_representations(thrust::device_vector<representation_t>& unique_representations_d,
                                               thrust::device_vector<std::uint32_t>& first_occurrence_index_d,
                                               const thrust::device_vector<representation_t>& input_representations_d);

/// \brief Helper kernel for find_first_occurrences_of_representations
///
/// Creates two arrays: first one contains a list of unique representations and the second one the index
/// at which that representation occurrs for the first time in the original data.
///
/// For example:
/// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
/// 0  0  0  0 12 12 12 12 12 12 23 23 23 32 32 32 32 32 46 46 46 <- input_representatons_d
/// 1  1  1  1  2  2  2  2  2  2  3  3  3  4  4  4  4  4  5  5  5 <- representation_index_mask_d
/// ^           ^                 ^        ^              ^
/// gives:
/// 0 12 23 32 46 <- unique_representations_d
/// 0  4 10 13 18 <- starting_index_of_each_representation_d
///
/// \param representation_index_mask_d an array in which each element from input_representatons_d is mapped to an ordinal number of that representation
/// \param input_representatons_d all representations
/// \param number_of_input_elements number of elements in input_representatons_d and representation_index_mask_d
/// \param starting_index_of_each_representation_d index with first occurrence of each representation
/// \param unique_representations_d representation that corresponds to each element in starting_index_of_each_representation_d
__global__ void find_first_occurrences_of_representations_kernel(const std::uint64_t* const representation_index_mask_d,
                                                                 const representation_t* const input_representations_d,
                                                                 const std::size_t number_of_input_elements,
                                                                 std::uint32_t* const starting_index_of_each_representation_d,
                                                                 representation_t* const unique_representations_d);

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

/// \brief Compresses unique_data and number_of_sketch_elements_with_representation after determening which ones should be filtered out
///
/// For example:
/// 4 <- filtering_threshold
/// 1  3  5  6  7    <- unique_representations_before_compression_d
/// 2  2  4  6  3  0 <- number_of_sketch_elements_with_representation_d (before filtering)
/// 2  2  0  0  3  0 <- number_of_sketch_elements_with_representation_d (after filtering)
/// 0  2  4  4  4  7 <- first_occurrence_of_representation_before_compression_d
/// 1  1  0  0  1    <- keep_representation_mask_d
/// 0  1  2  2  2  3 <- new_unique_representation_index_d (keep_representation_mask_d after exclusive sum)
///
/// after compression gives:
/// 1 3 7   <- unique_representations_after_compression_d
/// 0 2 4 7 <- first_occurrence_of_representation_after_compression_d
///
/// \param number_of_unique_representation_before_compression
/// \param unique_representations_before_compression_d
/// \param first_occurrence_of_representation_before_compression_d
/// \param new_unique_representation_index_d
/// \param unique_representations_after_compression_d
/// \param first_occurrence_of_representation_after_compression_d
__global__ void compress_unique_representations_after_filtering_kernel(const std::uint64_t number_of_unique_representation_before_compression,
                                                                       const representation_t* const unique_representations_before_compression_d,
                                                                       const std::uint32_t* const first_occurrence_of_representation_before_compression_d,
                                                                       const std::uint32_t* const new_unique_representation_index_d,
                                                                       representation_t* const unique_representations_after_compression_d,
                                                                       std::uint32_t* const first_occurrence_of_representation_after_compression_d);

/// \brief Compress sketch element data after determening which representations should be filtered out
///
/// For example:
/// 4 <- filtering_threshold
/// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
/// 1  1  3  3  5  5  5  5  6  6  6  6  6  6  7  7  7 <- representations_before_compression_d
/// 0  1  3  5  3  4  6  6  0  1  2  2  2  3  7  8  9 <- read_ids_before_compression_d
/// 0  0  1  1  4  5  8  9  3  6  7  8  9  5  4  7  3 <- positions_in_reads_before_compression_d
/// F  F  F  F  R  R  R  F  R  F  F  R  R  F  F  R  R <- directions_of_reads_before_compression_d
/// 1  3  5  6  7    <- unique_representations_before_compression_d
/// 2  2  4  6  3  0 <- number_of_sketch_elements_with_representation_d (before filtering)
/// 0  2  4  8 14 17 <- first_occurrence_of_representation_before_filtering_d
/// 2  2  0  0  3  0 <- number_of_sketch_elements_with_representation_before_compression_d (after filtering)
/// 0  2  4  4  4  7 <- first_occurrence_of_representation_before_compression_d (after filtering)
/// 0  2  4  7       <- first_occurrence_of_representation_after_compression_d
/// 1  1  0  0  1    <- keep_representation_mask_d
/// 0  1  2  2  2  3 <- unique_representation_index_after_compression_d (keep_representation_mask_d after exclusive sum)
///
/// after compression gives:
/// 0  1  2  3  4  5  6  7
/// 1  1  3  3  7  7  7    <- representations_after_compression_d
/// 0  1  3  5  7  8  9    <- read_ids_after_compression_d
/// 0  0  1  1  4  7  3    <- positions_in_reads_after_compression_d
/// F  F  F  F  F  R  R    <- directions_of_reads_after_compression_d
///
/// Launch with one thread block per unique representation, preferably with low number of threads per block
///
/// \param number_of_elements_before_compression
/// \param number_of_sketch_elements_with_representation_before_compression_d
/// \param first_occurrence_of_representation_before_filtering_d
/// \param first_occurrence_of_representation_after_compression_d
/// \param new_unique_representation_index_d
/// \param representations_before_compression_d
/// \param read_ids_before_compression_d
/// \param positions_in_reads_before_compression_d
/// \param directions_of_representations_before_compression_d
/// \param representations_after_compression_d
/// \param read_ids_after_compression_d
/// \param positions_in_reads_after_compression_d
/// \param directions_of_representations_after_compression_d
///
/// \tparam DirectionOfRepresentation any implementation of SketchElementImpl::SketchElementImpl::DirectionOfRepresentation
template <typename DirectionOfRepresentation>
__global__ void compress_data_arrays_after_filtering_kernel(const std::uint64_t number_of_unique_representations,
                                                            const std::uint32_t* const number_of_sketch_elements_with_representation_before_compression_d,
                                                            const std::uint32_t* const first_occurrence_of_representation_before_filtering_d,
                                                            const std::uint32_t* const first_occurrence_of_representation_after_compression_d,
                                                            const std::uint32_t* const unique_representation_index_after_compression_d,
                                                            const representation_t* const representations_before_compression_d,
                                                            const read_id_t* const read_ids_before_compression_d,
                                                            const position_in_read_t* const positions_in_reads_before_compression_d,
                                                            const DirectionOfRepresentation* const directions_of_representations_before_compression_d,
                                                            representation_t* const representations_after_compression_d,
                                                            read_id_t* const read_ids_after_compression_d,
                                                            position_in_read_t* const positions_in_reads_after_compression_d,
                                                            DirectionOfRepresentation* const directions_of_representations_after_compression_d)
{
    // TODO: investigate if launching one block per representation is a good idea
    // Ideally one would launch one thread pre sketch element, but that would requiry a binary search to determine corresponding
    // first_occurrence_of_representation_before_compression_d and others.
    // Another alternative would be to launch one thread per representation, but then it would happen that some threads in a block
    // would have way more work because there are more sketch elements with that representation.
    // Current solution works well if the average number of sketch elements per representation is equal or larger than the number of
    // threads per block. Otherwise a lot of blocks would end up with idle threads. That is why it is important to launch block with
    // small number of threads, ideally 32.

    const std::uint64_t representation_index_before_compression = blockIdx.x;

    if (representation_index_before_compression >= number_of_unique_representations)
        return;

    const std::uint32_t sketch_elements_with_this_representation = number_of_sketch_elements_with_representation_before_compression_d[representation_index_before_compression];

    if (0 == sketch_elements_with_this_representation) // this representation was filtered out
        return;

    const std::uint32_t first_occurrence_index_before_filtering  = first_occurrence_of_representation_before_filtering_d[representation_index_before_compression];
    const std::uint32_t first_occurrence_index_after_compression = first_occurrence_of_representation_after_compression_d[unique_representation_index_after_compression_d[representation_index_before_compression]];

    // now move all elements with that representation to the copressed array
    for (std::uint64_t i = threadIdx.x; i < sketch_elements_with_this_representation; i += blockDim.x)
    {
        representations_after_compression_d[first_occurrence_index_after_compression + i]               = representations_before_compression_d[first_occurrence_index_before_filtering + i];
        read_ids_after_compression_d[first_occurrence_index_after_compression + i]                      = read_ids_before_compression_d[first_occurrence_index_before_filtering + i];
        positions_in_reads_after_compression_d[first_occurrence_index_after_compression + i]            = positions_in_reads_before_compression_d[first_occurrence_index_before_filtering + i];
        directions_of_representations_after_compression_d[first_occurrence_index_after_compression + i] = directions_of_representations_before_compression_d[first_occurrence_index_before_filtering + i];
    }
}

/// \brief removes sketch elements with most common representations from the index
///
/// All sketch elements for which holds sketch_elementes_with_that_representation/total_sketch_element >= filtering_parameter will get removed
///
/// For example this index initinally contains 20 sketch elements:
/// 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
/// 1  1  3  3  5  5  5  5  6  6  6  6  6  6  7  7  7  8  8  8 <- representations (before filtering)
/// 0  1  3  5  3  4  6  6  0  1  2  2  2  3  7  8  9  1  2  3 <- read_ids (before filtering)
/// 0  0  1  1  4  5  8  9  3  6  7  8  9  5  4  7  3  7  8  9 <- positions_in_reads (before filtering)
/// F  F  F  F  R  R  R  F  R  F  F  R  R  F  F  R  R  F  F  F <- directions_of_reads (before filtering)
/// 1  3  5  6  7  8    <- unique_representations (before filtering)
/// 0  2  4  8 14 17 20 <- first_occurrence_of_representations (before filtering)
///
/// For filtering_parameter = 0.2:
/// sketch_elementes_with_that_representation/total_sketch_element >= 0.2 <=>
/// sketch_elementes_with_that_representation/20 >= 0.2 <=>
/// sketch_elementes_with_that_representation >= 20 * 0.2 <=>
/// sketch_elementes_with_that_representation >= 4 <=>
/// sketch element with representations with 4 or more sketch elements will be removed
///
/// In the example above that means that representations 5 and 6 will be removed and that the output would be:
/// 0  1  2  3  4  5  6  7  8  9
/// 1  1  3  3  7  7  7  8  8  8 <- representations (after filtering)
/// 0  1  3  5  7  8  9  1  2  3 <- read_ids (after filtering)
/// 0  0  1  1  4  7  3  7  8  9 <- positions_in_reads (after filtering)
/// F  F  F  F  F  R  R  F  F  F <- directions_of_reads (after filtering)
/// 1  3  7  8    <- unique_representations (after filtering)
/// 0  2  4  7 10 <- first_occurrence_of_representations (after filtering)
///
/// \param filtering_parameter value between 0 and 1'000'000'000, as explained above
/// \param representations_d original values on input, filtered on output
/// \param read_ids_d original values on input, filtered on output
/// \param positions_in_reads_d original values on input, filtered on output
/// \param directions_of_representations_d original values on input, filtered on output
/// \param unique_representation_d original values on input, filtered on output
/// \param first_occurrence_of_representations_d original values on input, filtered on output
///
/// \tparam DirectionOfRepresentation any implementation of SketchElementImpl::SketchElementImpl::DirectionOfRepresentation
template <typename DirectionOfRepresentation>
void filter_out_most_common_representations(const double filtering_parameter,
                                            thrust::device_vector<representation_t>& representations_d,
                                            thrust::device_vector<read_id_t>& read_ids_d,
                                            thrust::device_vector<position_in_read_t>& positions_in_reads_d,
                                            thrust::device_vector<DirectionOfRepresentation>& directions_of_representations_d,
                                            thrust::device_vector<representation_t>& unique_representations_d,
                                            thrust::device_vector<std::uint32_t>& first_occurrence_of_representations_d)
{
    // *** find the number of sketch_elements for every representation ***
    // 0  2  4  8 14 17 20 <- first_occurrence_of_representations_d (before filtering)
    // 2  2  4  6  3  3  0 <- number_of_sketch_elements_with_each_representation_d (with additional 0 at the end)

    thrust::device_vector<std::uint32_t> number_of_sketch_elements_with_each_representation_d(first_occurrence_of_representations_d.size());
    number_of_sketch_elements_with_each_representation_d.back() = 0; // H2D

    // thrust::adjacent_difference saves a[i]-a[i-1] to a[i]. As first_occurrence_of_representations_d starts with 0
    // we actually want to save a[i]-a[i-1] to a[i-j] and have the last (aditional) element of number_of_sketch_elements_with_each_representation_d set to 0
    thrust::adjacent_difference(thrust::device,
                                std::next(std::begin(first_occurrence_of_representations_d)),
                                std::end(first_occurrence_of_representations_d),
                                std::begin(number_of_sketch_elements_with_each_representation_d));
    number_of_sketch_elements_with_each_representation_d.back() = 0; // H2D

    // *** find filtering threshold ***
    const std::size_t total_sketch_elements = representations_d.size();
    // + 0.001 is a hacky workaround for problems which may arise when multiplying doubles and then casting into int
    const std::uint64_t filtering_threshold = static_cast<std::uint64_t>(total_sketch_elements * filtering_parameter + 0.001);

    // *** mark representations for filtering out ***
    // If representation is to be filtered out change its number of occurrences in number_of_sketch_elements_with_each_representation_d to 0
    // 2  2  4  6  3  3  0 <- number_of_sketch_elements_with_each_representation_d (before filtering)
    // 2  2  0  0  3  3  0 <- number_of_sketch_elements_with_each_representation_d (after filtering)

    thrust::replace_if(thrust::device,
                       std::begin(number_of_sketch_elements_with_each_representation_d),
                       std::prev(std::end(number_of_sketch_elements_with_each_representation_d)), // don't process the last element, it should remain 0
                       [filtering_threshold] __device__(const std::uint32_t val) {
                           return val >= filtering_threshold;
                       },
                       0);

    // *** perform exclusive sum to find the starting position of each representation after filtering ***
    // If a representation is to be filtered out its value is the same to the value to its left
    // 2  2  0  0  3  3  0 <- number_of_sketch_elements_with_each_representation_d (after filtering)
    // 0  2  4  4  4  7 10 <- first_occurrence_of_representation_after_filtering_d
    thrust::device_vector<std::uint32_t> first_occurrence_of_representation_after_filtering_d(number_of_sketch_elements_with_each_representation_d.size());
    thrust::exclusive_scan(thrust::device,
                           std::begin(number_of_sketch_elements_with_each_representation_d),
                           std::end(number_of_sketch_elements_with_each_representation_d),
                           std::begin(first_occurrence_of_representation_after_filtering_d));

    // *** create unique_representation_index_after_filtering_d ***
    // unique_representation_index_after_filtering_d contains the index of that representation after filtering if the representation is going to be kept
    // or the value of its left neighbor if the representation is going to be filtered out.
    // Additional element at the end contains the total number of unique representations after filtering
    // 2  2  0  0  3  3  0 <- number_of_sketch_elements_with_each_representation_d (after filtering)
    // 1  1  0  0  1  1  0 <- helper array with 1 if representation is going to be kept and 0 otherwise
    // 0  1  2  2  2  3  4 <- unique_representation_index_after_filtering_d

    const std::int64_t number_of_unique_representations = get_size(unique_representations_d);
    thrust::device_vector<std::uint32_t> unique_representation_index_after_filtering_d(number_of_unique_representations + 1);

    {
        // direct pointer needed as device_vector::operator[] is a host function and it would be called from device lambda
        const std::uint32_t* const number_of_sketch_elements_with_each_representation_d_ptr = number_of_sketch_elements_with_each_representation_d.data().get();
        thrust::transform_exclusive_scan(thrust::device,
                                         thrust::make_counting_iterator(std::int64_t(0)),
                                         thrust::make_counting_iterator(number_of_unique_representations + 1),
                                         std::begin(unique_representation_index_after_filtering_d),
                                         [number_of_sketch_elements_with_each_representation_d_ptr, number_of_unique_representations] __device__(const std::int64_t unique_representation_index) -> std::uint32_t {
                                             if (unique_representation_index < number_of_unique_representations)
                                                 return (0 == number_of_sketch_elements_with_each_representation_d_ptr[unique_representation_index] ? 0 : 1);
                                             else // the additional element at the end
                                                 return 0;
                                         },
                                         0,
                                         thrust::plus<std::uint64_t>());
    }

    // *** remove filtered out elements (compress) from unique_representations_d and first_occurrence_of_representations_d ***
    // 1  3  5  6  7  8 <- unique_representations_d (original)
    // 1  3  7  8       <- unique_representations_after_filtering_d
    //
    // 0  2  4  4  4  7 10 <- first_occurrence_of_representation_after_filtering_d
    // 0  2  4  7 10       <- first_occurrence_of_representations_after_compression_d

    const std::uint32_t number_of_unique_representations_after_compression = unique_representation_index_after_filtering_d.back(); // D2H

    thrust::device_vector<representation_t> unique_representations_after_compression_d(number_of_unique_representations_after_compression);
    thrust::device_vector<std::uint32_t> first_occurrence_of_representations_after_compression_d(number_of_unique_representations_after_compression + 1);

    std::int32_t number_of_threads = 128;
    std::int32_t number_of_blocks  = ceiling_divide<std::int64_t>(first_occurrence_of_representations_d.size(),
                                                                 number_of_threads);

    compress_unique_representations_after_filtering_kernel<<<number_of_blocks, number_of_threads>>>(unique_representations_d.size(),
                                                                                                    unique_representations_d.data().get(),
                                                                                                    first_occurrence_of_representation_after_filtering_d.data().get(),
                                                                                                    unique_representation_index_after_filtering_d.data().get(),
                                                                                                    unique_representations_after_compression_d.data().get(),
                                                                                                    first_occurrence_of_representations_after_compression_d.data().get());

    // *** remove filtered out elements (compress) from other data arrays ***

    // 1  1  3  3  5  5  5  5  6  6  6  6  6  6  7  7  7  8  8  8 <- representations_d (before filtering)
    // 0  1  3  5  3  4  6  6  0  1  2  2  2  3  7  8  9  1  2  3 <- read_ids_d (before filtering)
    // 0  0  1  1  4  5  8  9  3  6  7  8  9  5  4  7  3  7  8  9 <- positions_in_reads_d (before filtering)
    // F  F  F  F  R  R  R  F  R  F  F  R  R  F  F  R  R  F  F  F <- directions_of_reads_d (before filtering)
    // 1  1  3  3  7  7  7  8  8  8 <- representations_after_filtering_d
    // 0  1  3  5  7  8  9  1  2  3 <- read_ids_after_filtering_d
    // 0  0  1  1  4  7  3  7  8  9 <- positions_in_reads_after_filtering_d
    // F  F  F  F  F  R  R  F  F  F <- directions_of_reads_after_filtering_d

    std::uint32_t number_of_sketch_elements_after_compression = first_occurrence_of_representations_after_compression_d.back(); // D2H
    thrust::device_vector<representation_t> representations_after_compression_d(number_of_sketch_elements_after_compression);
    thrust::device_vector<read_id_t> read_ids_after_compression_d(number_of_sketch_elements_after_compression);
    thrust::device_vector<position_in_read_t> positions_in_reads_after_compression_d(number_of_sketch_elements_after_compression);
    thrust::device_vector<DirectionOfRepresentation> directions_of_representations_after_compression_d(number_of_sketch_elements_after_compression);

    number_of_threads = 32;
    number_of_blocks  = unique_representations_d.size();
    compress_data_arrays_after_filtering_kernel<<<number_of_blocks, number_of_threads>>>(unique_representations_d.size(),
                                                                                         number_of_sketch_elements_with_each_representation_d.data().get(),
                                                                                         first_occurrence_of_representations_d.data().get(),
                                                                                         first_occurrence_of_representations_after_compression_d.data().get(),
                                                                                         unique_representation_index_after_filtering_d.data().get(),
                                                                                         representations_d.data().get(),
                                                                                         read_ids_d.data().get(),
                                                                                         positions_in_reads_d.data().get(),
                                                                                         directions_of_representations_d.data().get(),
                                                                                         representations_after_compression_d.data().get(),
                                                                                         read_ids_after_compression_d.data().get(),
                                                                                         positions_in_reads_after_compression_d.data().get(),
                                                                                         directions_of_representations_after_compression_d.data().get());

    // *** swap vectors with the input arrays ***
    std::swap(unique_representations_d, unique_representations_after_compression_d);
    std::swap(first_occurrence_of_representations_d, first_occurrence_of_representations_after_compression_d);
    std::swap(representations_d, representations_after_compression_d);
    std::swap(read_ids_d, read_ids_after_compression_d);
    std::swap(positions_in_reads_d, positions_in_reads_after_compression_d);
    std::swap(directions_of_representations_d, directions_of_representations_after_compression_d);
}

} // namespace index_gpu
} // namespace details

template <typename SketchElementImpl>
IndexGPU<SketchElementImpl>::IndexGPU(const io::FastaParser& parser,
                                      const read_id_t first_read_id,
                                      const read_id_t past_the_last_read_id,
                                      const std::uint64_t kmer_size,
                                      const std::uint64_t window_size,
                                      const bool hash_representations,
                                      const double filtering_parameter)
    : first_read_id_(first_read_id)
    , kmer_size_(kmer_size)
    , window_size_(window_size)
    , number_of_reads_(0)
{
    generate_index(parser,
                   first_read_id_,
                   past_the_last_read_id,
                   hash_representations,
                   filtering_parameter);
}

template <typename SketchElementImpl>
const thrust::device_vector<representation_t>& IndexGPU<SketchElementImpl>::representations() const
{
    return representations_d_;
};

template <typename SketchElementImpl>
const thrust::device_vector<read_id_t>& IndexGPU<SketchElementImpl>::read_ids() const
{
    return read_ids_d_;
}

template <typename SketchElementImpl>
const thrust::device_vector<position_in_read_t>& IndexGPU<SketchElementImpl>::positions_in_reads() const
{
    return positions_in_reads_d_;
}

template <typename SketchElementImpl>
const thrust::device_vector<typename SketchElementImpl::DirectionOfRepresentation>& IndexGPU<SketchElementImpl>::directions_of_reads() const
{
    return directions_of_reads_d_;
}

template <typename SketchElementImpl>
const thrust::device_vector<representation_t>& IndexGPU<SketchElementImpl>::unique_representations() const
{
    return unique_representations_d_;
}

template <typename SketchElementImpl>
const thrust::device_vector<std::uint32_t>& IndexGPU<SketchElementImpl>::first_occurrence_of_representations() const
{
    return first_occurrence_of_representations_d_;
}

template <typename SketchElementImpl>
const std::string& IndexGPU<SketchElementImpl>::read_id_to_read_name(const read_id_t read_id) const
{
    return read_id_to_read_name_[read_id - first_read_id_];
}

template <typename SketchElementImpl>
const std::uint32_t& IndexGPU<SketchElementImpl>::read_id_to_read_length(const read_id_t read_id) const
{
    return read_id_to_read_length_[read_id - first_read_id_];
}

template <typename SketchElementImpl>
std::uint64_t IndexGPU<SketchElementImpl>::number_of_reads() const
{
    return number_of_reads_;
}

template <typename SketchElementImpl>
void IndexGPU<SketchElementImpl>::generate_index(const io::FastaParser& parser,
                                                 const read_id_t first_read_id,
                                                 const read_id_t past_the_last_read_id,
                                                 const bool hash_representations,
                                                 const double filtering_parameter)
{

    // check if there are any reads to process
    if (first_read_id >= past_the_last_read_id)
    {
        CGA_LOG_INFO("No Sketch Elements to be added to index");
        number_of_reads_ = 0;
        return;
    }

    number_of_reads_ = 0;

    std::uint64_t total_basepairs = 0;
    std::vector<ArrayBlock> read_id_to_basepairs_section_h;
    std::vector<io::FastaSequence> fasta_reads;

    // deterine the number of basepairs in each read and assign read_id to each read
    for (read_id_t read_id = first_read_id; read_id < past_the_last_read_id; ++read_id)
    {
        fasta_reads.emplace_back(parser.get_sequence_by_id(read_id));
        const std::string& read_basepairs = fasta_reads.back().seq;
        const std::string& read_name      = fasta_reads.back().name;
        if (read_basepairs.length() >= window_size_ + kmer_size_ - 1)
        {
            read_id_to_basepairs_section_h.emplace_back(ArrayBlock{total_basepairs, static_cast<std::uint32_t>(read_basepairs.length())});
            total_basepairs += read_basepairs.length();
            read_id_to_read_name_.push_back(read_name);
            read_id_to_read_length_.push_back(read_basepairs.length());
            number_of_reads_++;
        }
        else
        {
            CGA_LOG_INFO("Skipping read {}. It has {} basepairs, one window covers {} basepairs",
                         read_name,
                         read_basepairs.length(),
                         window_size_ + kmer_size_ - 1);
            // Remove the most recent sequence from the reads vector as it isn't part of the index.
            fasta_reads.pop_back();
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
                                                                       read_id_to_basepairs_section_d,
                                                                       hash_representations);
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
    if (representations_d_.capacity() > representations_d_.size())
        representations_d_.shrink_to_fit();
    thrust::copy(thrust::device,
                 representations_d.data(),
                 representations_d.data() + representations_d.size(),
                 representations_d_.begin());
    representations_d.free();

    read_ids_d_.resize(representations_d_.size());
    if (read_ids_d_.capacity() > read_ids_d_.size())
        read_ids_d_.shrink_to_fit();
    positions_in_reads_d_.resize(representations_d_.size());
    if (positions_in_reads_d_.capacity() > positions_in_reads_d_.size())
        positions_in_reads_d_.shrink_to_fit();
    directions_of_reads_d_.resize(representations_d_.size());
    if (directions_of_reads_d_.capacity() > directions_of_reads_d_.size())
        directions_of_reads_d_.shrink_to_fit();

    const std::uint32_t threads = 256;
    const std::uint32_t blocks  = ceiling_divide<int64_t>(representations_d_.size(), threads);

    details::index_gpu::copy_rest_to_separate_arrays<<<blocks, threads>>>(rest_d.data(),
                                                                          read_ids_d_.data().get(),
                                                                          positions_in_reads_d_.data().get(),
                                                                          directions_of_reads_d_.data().get(),
                                                                          representations_d_.size());

    // now generate the index elements
    details::index_gpu::find_first_occurrences_of_representations(unique_representations_d_,
                                                                  first_occurrence_of_representations_d_,
                                                                  representations_d_);

    if (filtering_parameter < 1.0)
    {
        details::index_gpu::filter_out_most_common_representations(filtering_parameter,
                                                                   representations_d_,
                                                                   read_ids_d_,
                                                                   positions_in_reads_d_,
                                                                   directions_of_reads_d_,
                                                                   unique_representations_d_,
                                                                   first_occurrence_of_representations_d_);
    }
}

} // namespace cudamapper
} // namespace claragenomics
