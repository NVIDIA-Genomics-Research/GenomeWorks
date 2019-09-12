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

#include <algorithm>
#include <exception>
#include <numeric>
#include <utility>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <claragenomics/logging/logging.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/device_buffer.cuh>

#include "cudamapper/index.hpp"
#include "cudamapper/types.hpp"

#include "bioparser/bioparser.hpp"
#include "bioparser_sequence.hpp"

#include "cudamapper_utils.hpp"

namespace claragenomics {

    /// IndexGPU - Contains sketch elements grouped by representation and by read id within the representation
    ///
    /// Class contains three separate data arrays: read_ids, positions_in_reads and directions_of_reads.
    /// Elements of these three arrays with the same index represent one sketch element
    /// (read_id of the read it belongs to, position in that read of the first basepair of sketch element and whether it is forward or reverse complement representation).
    /// Representation itself is not saved as it is not necessary for matching phase. It can be retrieved from the original data if needed.
    ///
    /// Elements of data arrays are grouped by sketch element representation and within those groups by read_id. Both representations and read_ids within representations are sorted in ascending order
    ///
    /// read_id_and_representation_to_sketch_elements() for each read_id (outer vector) returns a vector in which each element contains a representation from that read, pointer to section of data arrays with sketch elements with that representation and that read_id, and pointer to section of data arrays with skecth elements with that representation and all read_ids. There elements are sorted by representation in increasing order
    ///
    /// \tparam SketchElementImpl any implementation of SketchElement
    template <typename SketchElementImpl>
    class IndexGPU : public Index {
    public:

        /// \brief Constructor
        ///
        /// \param query_filename filepath to reads in FASTA or FASTQ format
        /// \param kmer_size k - the kmer length
        /// \param window_size w - the length of the sliding window used to find sketch elements
        IndexGPU(const std::string& query_filename, const std::uint64_t kmer_size, const std::uint64_t window_size);

        /// \brief Constructor
        IndexGPU();

        /// \brief returns an array of starting positions of sketch elements in their reads
        /// \return an array of starting positions of sketch elements in their reads
        const std::vector<position_in_read_t>& positions_in_reads() const override;

        /// \brief returns an array of reads ids for sketch elements
        /// \return an array of reads ids for sketch elements
        const std::vector<read_id_t>& read_ids() const override;

        /// \brief returns an array of directions in which sketch elements were read
        /// \return an array of directions in which sketch elements were read
        const std::vector<typename SketchElementImpl::DirectionOfRepresentation>& directions_of_reads() const override;

        /// \brief returns number of reads in input data
        /// \return number of reads in input data
        std::uint64_t number_of_reads() const override;

        /// \brief returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        /// \return mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        const std::vector<std::string>& read_id_to_read_name() const override;

        /// \brief returns mapping of internal read id that goes from 0 to read lengths for that read
        /// \return mapping of internal read id that goes from 0 to read lengths for that read
        const std::vector<std::uint32_t>& read_id_to_read_length() const override;

        /// \brief For each read_id (outer vector) returns a vector in which each element contains a representation from that read, pointer to section of data arrays with sketch elements with that representation and that read_id, and pointer to section of data arrays with skecth elements with that representation and all read_ids. There elements are sorted by representation in increasing order
        /// \return the mapping
        const std::vector<std::vector<Index::RepresentationToSketchElements>>& read_id_and_representation_to_sketch_elements() const override;

    private:

        /// \brief generates the index
        /// \param query_filename
        void generate_index(const std::string& query_filename);

        const std::uint64_t kmer_size_;
        const std::uint64_t window_size_;
        std::uint64_t number_of_reads_;

        std::vector<position_in_read_t> positions_in_reads_;
        std::vector<read_id_t> read_ids_;
        std::vector<typename SketchElementImpl::DirectionOfRepresentation> directions_of_reads_;

        std::vector<std::string> read_id_to_read_name_;
        std::vector<std::uint32_t> read_id_to_read_length_;

        std::vector<std::vector<RepresentationToSketchElements>> read_id_and_representation_to_sketch_elements_;
    };

namespace details {

namespace index_gpu {

    /// approximate_sketch_elements_per_bucket_too_short - exception thrown when the number of sketch_elements_per_bucket is too small
    class approximate_sketch_elements_per_bucket_too_small : public std::exception
    {
    public:
        approximate_sketch_elements_per_bucket_too_small(const std::string& message)
        : message_(message) {}
        approximate_sketch_elements_per_bucket_too_small(approximate_sketch_elements_per_bucket_too_small const&)            = default;
        approximate_sketch_elements_per_bucket_too_small& operator=(approximate_sketch_elements_per_bucket_too_small const&) = default;
        virtual ~approximate_sketch_elements_per_bucket_too_small()                                                          = default;

        virtual const char* what() const noexcept override
        {
            return message_.data();
        }
    private:
        const std::string message_;
    };

    /// @brief Takes multiple arrays of sketch elements and determines an array of representations such that the number of elements between each two representations is similar to the given value
    ///
    /// Function takes multiple arrays of sketch elements. Elements of each array are sorted by representation
    /// The function generates an array of representations such that if all input arrays were sorted together the number of sketch elements
    /// between neighboring elements would not be similar to approximate_sketch_elements_per_bucket.
    /// The number of element in a bucket is guaranteed to be <= approximate_sketch_elements_per_bucket, unless members_with_some_representation >= approximate_sketch_elements_per_bucket,
    /// in which case the number of elements in its bucket is guaranteed to be  <= members_with_that_representation + approximate_sketch_elements_per_bucket (this is not expect with genomes as
    /// approximate_sketch_elements_per_bucket should be the number of elements that can fit one GPU).
    /// All elements with the same representation are guaranteed to be in the same bucket
    ///
    /// Take the following three arrays and approximate_sketch_elements_per_bucket = 5:
    /// (0 1 2 3 3 5 6 7 7 9) <- index
    /// (1 1 2 2 4 4 6 6 9 9)
    /// (0 0 1 5 5 5 7 8 8 8)
    /// (1 1 1 3 3 4 5 7 9 9)
    ///
    /// When all three arrays are merged and sorte this give:
    /// (0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29)
    /// (0  0  1  1  1  1  1  1  2  2  3  3  3  4  4  5  5  5  5  6  6  7  7  7  8  8  9  9  9  9)
    ///  ^     ^                 ^              ^     ^           ^              ^     ^
    ///
    /// Representation in the output array and the respective chunks would be:
    /// 0: 0 0
    /// 1: 1 1 1 1 1 1 <- larger the approximate_sketch_elements_per_bucket, so only one representation in this chunk
    /// 2: 2 2 3 3 3
    /// 4: 4 4
    /// 5: 5 5 5 5
    /// 6: 6 6 7 7 7
    /// 8: 8 8
    /// 9: 9 9 9 9
    ///
    /// Note that the line 1 could also be "1 1 1 1 1 1 2 2" or  "1 1 1 1 1 1 2 2 3 3 3", but not "1 1 1 1 1 1 2 2 3 3 3 4 4"
    ///
    /// \param arrays_of_representations multiple arrays of sketch element representations in which elements are sorted by representation
    /// \param approximate_sketch_elements_per_bucket approximate number of sketch elements between two representations
    /// \return list of representations that limit the buckets (left boundary inclusive, right exclusive)
    /// \throw approximate_sketch_elements_per_bucket_too_small if approximate_sketch_elements_per_bucket is too small
    std::vector<representation_t> generate_representation_buckets(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                                                  const std::uint64_t approximate_sketch_elements_per_bucket
                                                                 );

    /// \brief Gets the index of first occurrence of the given representation in each array
    ///
    /// \param arrays_of_representations multiple arrays of sketch element representations in which elements are sorted by representation
    /// \param representation representation to look for
    /// \return for each array in arrays_of_representations contains the index for the first element greater or equal to representation, or the index of past-the-last element if all elements have a smaller representation
    std::vector<std::size_t> generate_representation_indices(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                                             const representation_t representation
                                                            );

    /// \brief For each bucket generates first and past-the-last index that fall into that bucket for each array of representations
    ///
    /// \param arrays_of_representations multiple arrays of sketch element representations in which elements are sorted by representation
    /// \param generate_representation_buckets first reprentation in each bucket
    /// \return outer vector goes over all buckets, inner gives first and past-the-last index of that bucket in every array of representations, if first and past-the-last index are the same that means that there are no elements of that bucket in that array
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> generate_bucket_boundary_indices(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                                                                                   const std::vector<representation_t>& representation_buckets
                                                                                                  );

    /// \brief Takes multiple arrays of sketch elements and merges them together so that the output array is sorted
    ///
    /// Function takes multiple arrays of sketch elements (each of those arrays is actually split into two arrays, one containing representations
    /// and the other read ids, positions in reads and directions, but for the sake of simplicity they are treated as one array in comments).
    /// Sketch elements in each input array are sorted by representation.
    /// On the output all arrays are merged in one big array and sorted by representations.
    /// Within each representation group the order of the elements from arrays_of_readids_positions_directions remains the same and they are ordered
    /// by the index of their array in arrays_of_readids_positions_directions, i.e. first come elements from arrays_of_readids_positions_directions[0],
    /// than arrays_of_readids_positions_directions[1]...
    ///
    /// \param arrays_of_representations multiple arrays of sketch element representations in which elements are sorted by representation
    /// \param arrays_of_readids_positions_directions multiple arrays of sketch elements (excluding representation) in which elements are sorted by representation
    /// \param available_device_memory_bytes how much GPU memory is available for this merge
    /// \param merged_representations on output contains all sketch element representations from all arrays_of_representations_h subarrays, sorted by representation and further sorted by read_id within each representation group
    /// \param merged_readids_positions_directions contains all sketch elements (excluding representation) from all arrays_of_readid_position_direction_h subarrays, sorted by representation and further sorted by read_id within each representation group
    ///
    /// \tparam ReadidPositionDirection any implementation of SketchElement::ReadidPositionDirection
    template <typename ReadidPositionDirection>
    void merge_sketch_element_arrays(const std::vector<std::vector<representation_t>>& arrays_of_representations,
                                     const std::vector<std::vector<ReadidPositionDirection>>& arrays_of_readids_positions_directions,
                                     const std::uint64_t available_device_memory_bytes,
                                     std::vector<representation_t>& merged_representations,
                                     std::vector<ReadidPositionDirection>& merged_readids_positions_directions)
    {
        // Each array in arrays_of_representations (and arrays_of_readids_positions_directions) is sorted by representation.
        // The goal is to have the data from all arrays merged together. If data from all arrays fits the device memory this can be done by copying from all arrays to one
        // device array, sorting it on device and copying it back to host.
        // If the data is too large merging has to be done in chunks. As the data in arrays is already sorted it is possible to take the data for all representations
        // between rep_x and rep_y from all arrays, put it on device, sort and move the sorted data back to host.
        // If these chunks are chosen as ((rep_0, rep_x), (rep_x + 1, rep_y), (rep_y + 1, rep_z) ...) the final result will be completely sorted.
        // generate_bucket_boundary_indices generates buckets/chunks of representations so that they fit the device memory

        std::uint64_t size_of_one_element = sizeof(representation_t) + sizeof(ReadidPositionDirection);
        // how many elements can be sorted at once (thrust::stable_sort_by_key is done out-of-place, hence 2.1)
        std::uint64_t elements_per_merge = ( (available_device_memory_bytes / 21 ) * 10 ) / size_of_one_element;

        // generate buckets
        std::vector<std::vector<std::pair<std::size_t, std::size_t>>> bucket_boundary_indices = generate_bucket_boundary_indices(arrays_of_representations,
                                                                                                                                 generate_representation_buckets(arrays_of_representations,
                                                                                                                                                                 elements_per_merge
                                                                                                                                                                )
                                                                                                                                );

        const std::size_t number_of_buckets = bucket_boundary_indices.size();
        const std::size_t number_of_arrays = arrays_of_representations.size();

        // find longest output bucket
        std::size_t longest_merged_bucket_length = 0;
        for (const auto& input_buckets_for_one_output_bucket : bucket_boundary_indices) {
            std::size_t length = 0;
            for (const auto& one_input_bucket : input_buckets_for_one_output_bucket) {
                length += one_input_bucket.second - one_input_bucket.first;
            }
            longest_merged_bucket_length = std::max(longest_merged_bucket_length, length);
        }

        // allocate the array that will be used for merging
        CGA_LOG_INFO("Allocating {} bytes for representations_bucket_to_merge_d", longest_merged_bucket_length * sizeof(representation_t));
        device_buffer<representation_t> representations_bucket_to_merge_d(longest_merged_bucket_length);
        CGA_LOG_INFO("Allocating {} bytes for readids_positions_directions_bucket_to_merge_d", longest_merged_bucket_length * sizeof(ReadidPositionDirection));
        device_buffer<ReadidPositionDirection> readids_positions_directions_bucket_to_merge_d(longest_merged_bucket_length);

        // find total number of sketch elements in all subarrays
        std::size_t total_sketch_elements = 0;
        total_sketch_elements = std::accumulate(std::begin(arrays_of_representations),
                                                std::end(arrays_of_representations),
                                                0,
                                                [](auto counter, const auto& one_array) { return counter += one_array.size(); }
                                               );

        // allocate enough space for merged sketch elements
        merged_representations.resize(total_sketch_elements);
        merged_readids_positions_directions.resize(total_sketch_elements);

        // go bucket by bucket
        std::size_t output_elements_written = 0;
        for (std::size_t bucket_index = 0; bucket_index < number_of_buckets; ++bucket_index) {
            // copy data from all arrays which belongs to that bucket
            std::size_t elements_written = 0;
            for (std::size_t array_index = 0; array_index < number_of_arrays; ++array_index) {
                std::size_t elements_to_copy = bucket_boundary_indices[bucket_index][array_index].second - bucket_boundary_indices[bucket_index][array_index].first;
                if (elements_to_copy > 0) {
                    // to reduce the number of cudaMemcpys one could do all copies to a host buffer and than copy all data to device at once, but that would take more space
                    CGA_CU_CHECK_ERR(cudaMemcpy(representations_bucket_to_merge_d.data() + elements_written,
                                                arrays_of_representations[array_index].data() + bucket_boundary_indices[bucket_index][array_index].first,
                                                elements_to_copy * sizeof(representation_t),
                                                cudaMemcpyHostToDevice
                                               )
                                    );
                    CGA_CU_CHECK_ERR(cudaMemcpy(readids_positions_directions_bucket_to_merge_d.data() + elements_written,
                                                arrays_of_readids_positions_directions[array_index].data() + bucket_boundary_indices[bucket_index][array_index].first,
                                                elements_to_copy * sizeof(ReadidPositionDirection),
                                                cudaMemcpyHostToDevice
                                               )
                                    );
                    elements_written += elements_to_copy;
                }
            }
            // sort bucket
            thrust::stable_sort_by_key(thrust::device,
                                       representations_bucket_to_merge_d.data(),
                                       representations_bucket_to_merge_d.data() + elements_written,
                                       readids_positions_directions_bucket_to_merge_d.data()
                                      );
            // copy sorted bucket to host output array
            CGA_CU_CHECK_ERR(cudaMemcpy(merged_representations.data() + output_elements_written,
                                        representations_bucket_to_merge_d.data(),
                                        elements_written * sizeof(representation_t),
                                        cudaMemcpyDeviceToHost
                                       )
                                    );
            CGA_CU_CHECK_ERR(cudaMemcpy(merged_readids_positions_directions.data() + output_elements_written,
                                        readids_positions_directions_bucket_to_merge_d.data(),
                                        elements_written * sizeof(ReadidPositionDirection),
                                        cudaMemcpyDeviceToHost
                                       )
                                    );
            output_elements_written += elements_written;
        }

        CGA_LOG_INFO("Deallocating {} bytes from representations_bucket_to_merge_d", longest_merged_bucket_length * sizeof(representation_t));
        representations_bucket_to_merge_d.free();
        CGA_LOG_INFO("Deallocating {} bytes from readids_positions_directions_bucket_to_merge_d", longest_merged_bucket_length * sizeof(ReadidPositionDirection));
        readids_positions_directions_bucket_to_merge_d.free();
    }

} // namespace index_gpu

} // namespace details

    template <typename SketchElementImpl>
    IndexGPU<SketchElementImpl>::IndexGPU(const std::string& query_filename, const std::uint64_t kmer_size, const std::uint64_t window_size)
    : kmer_size_(kmer_size), window_size_(window_size), number_of_reads_(0)
    {
        generate_index(query_filename);
    }

    template <typename SketchElementImpl>
    IndexGPU<SketchElementImpl>::IndexGPU()
    : kmer_size_(0), window_size_(0), number_of_reads_(0) {
    }

    template <typename SketchElementImpl>
    const std::vector<position_in_read_t>& IndexGPU<SketchElementImpl>::positions_in_reads() const { return positions_in_reads_; }

    template <typename SketchElementImpl>
    const std::vector<read_id_t>& IndexGPU<SketchElementImpl>::read_ids() const { return read_ids_; }

    template <typename SketchElementImpl>
    const std::vector<typename SketchElementImpl::DirectionOfRepresentation>& IndexGPU<SketchElementImpl>::directions_of_reads() const { return directions_of_reads_; }

    template <typename SketchElementImpl>
    std::uint64_t IndexGPU<SketchElementImpl>::number_of_reads() const { return number_of_reads_; }

    template <typename SketchElementImpl>
    const std::vector<std::string>& IndexGPU<SketchElementImpl>::read_id_to_read_name() const { return read_id_to_read_name_; }

    template <typename SketchElementImpl>
    const std::vector<std::uint32_t>& IndexGPU<SketchElementImpl>::read_id_to_read_length() const { return read_id_to_read_length_; }

    template <typename SketchElementImpl>
    const std::vector<std::vector<Index::RepresentationToSketchElements>>& IndexGPU<SketchElementImpl>::read_id_and_representation_to_sketch_elements() const { return read_id_and_representation_to_sketch_elements_; }

    // TODO: This function will be split into several functions
    template <typename SketchElementImpl>
    void IndexGPU<SketchElementImpl>::generate_index(const std::string& query_filename) {
        std::unique_ptr<bioparser::Parser<BioParserSequence>> query_parser = nullptr;

        auto is_suffix = [](const std::string &src, const std::string &suffix) -> bool {
            if (src.size() < suffix.size()) {
                return false;
            }
            return src.compare(src.size() - suffix.size(), suffix.size(), suffix) == 0;
        };

        if (is_suffix(query_filename, ".fasta") || is_suffix(query_filename, ".fa") ||
            is_suffix(query_filename, ".fasta.gz") || is_suffix(query_filename, ".fa.gz")) {
            CGA_LOG_INFO("Getting Query data");
            query_parser = bioparser::createParser<bioparser::FastaParser, BioParserSequence>(query_filename);
        }

        //TODO Allow user to choose this value
        std::uint64_t parser_buffer_size_in_bytes = 0.3 * 1024 * 1024 * 1024; // 0.3 GiB

        number_of_reads_ = 0;

        std::vector<std::vector<representation_t>> representations_from_all_loops_h;
        std::vector<std::vector<typename SketchElementImpl::ReadidPositionDirection>> rest_from_all_loops_h;

        while (true) {
            //read the query file:
            std::vector<std::unique_ptr<BioParserSequence>> fasta_objects;
            bool parser_status = query_parser->parse(fasta_objects, parser_buffer_size_in_bytes);

            std::uint64_t total_basepairs = 0;
            std::vector<ArrayBlock> read_id_to_basepairs_section_h;

            // find out how many basepairs each read has and determine its section in the big array with all basepairs
            for (std::size_t fasta_object_id = 0; fasta_object_id < fasta_objects.size(); ++fasta_object_id) {
                // skip reads which are shorter than one window
                if (fasta_objects[fasta_object_id]->data().length() >= window_size_ + kmer_size_ - 1) {
                    read_id_to_basepairs_section_h.emplace_back(ArrayBlock{total_basepairs, static_cast<std::uint32_t>(fasta_objects[fasta_object_id]->data().length())});
                    total_basepairs += fasta_objects[fasta_object_id]->data().length();
                    read_id_to_read_name_.push_back(fasta_objects[fasta_object_id]->name());
                    read_id_to_read_length_.push_back(fasta_objects[fasta_object_id]->data().length());
                } else {
                    CGA_LOG_INFO("Skipping read {}. It has {} basepairs, one window covers {} basepairs",
                                 fasta_objects[fasta_object_id]->name(),
                                 fasta_objects[fasta_object_id]->data().length(), window_size_ + kmer_size_ - 1
                                );
                }
            }

            auto number_of_reads_to_add = read_id_to_basepairs_section_h.size(); // This is the number of reads in this specific iteration
            number_of_reads_ += number_of_reads_to_add; // this is the *total* number of reads.

            //Stop if there are no reads to add
            if (0 == number_of_reads_to_add) {
                if (parser_status == false) {
                    break;
                }
                continue;
            }

            std::vector<char> merged_basepairs_h(total_basepairs);

            // copy each read to its section of the basepairs array
            read_id_t read_id = 0;
            for (std::size_t fasta_object_id = 0; fasta_object_id < number_of_reads_to_add; ++fasta_object_id) {
                // skip reads which are shorter than one window
                if (fasta_objects[fasta_object_id]->data().length() >= window_size_ + kmer_size_ - 1) {
                    std::copy(std::begin(fasta_objects[fasta_object_id]->data()),
                              std::end(fasta_objects[fasta_object_id]->data()),
                              std::next(std::begin(merged_basepairs_h), read_id_to_basepairs_section_h[read_id].first_element_)
                             );
                    ++read_id;
                }
            }

            // fasta_objects not needed after this point
            fasta_objects.clear();
            fasta_objects.reserve(0);

            // move basepairs to the device
            CGA_LOG_INFO("Allocating {} bytes for read_id_to_basepairs_section_d", read_id_to_basepairs_section_h.size() * sizeof(decltype(read_id_to_basepairs_section_h)::value_type));
            device_buffer<decltype(read_id_to_basepairs_section_h)::value_type> read_id_to_basepairs_section_d( read_id_to_basepairs_section_h.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(read_id_to_basepairs_section_d.data(),
                                        read_id_to_basepairs_section_h.data(),
                                        read_id_to_basepairs_section_h.size() * sizeof(decltype(read_id_to_basepairs_section_h)::value_type),
                                        cudaMemcpyHostToDevice
                                       )
                            );

            CGA_LOG_INFO("Allocating {} bytes for merged_basepairs_d", merged_basepairs_h.size() * sizeof(decltype(merged_basepairs_h)::value_type));
            device_buffer<decltype(merged_basepairs_h)::value_type> merged_basepairs_d(merged_basepairs_h.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(merged_basepairs_d.data(),
                                        merged_basepairs_h.data(),
                                        merged_basepairs_h.size() * sizeof(decltype(merged_basepairs_h)::value_type),
                                        cudaMemcpyHostToDevice
                                       )
                            );
            merged_basepairs_h.clear();
            merged_basepairs_h.reserve(0);

            // sketch elements get generated here
            auto sketch_elements = SketchElementImpl::generate_sketch_elements(number_of_reads_to_add,
                                                                                        kmer_size_,
                                                                                        window_size_,
                                                                                        number_of_reads_ - number_of_reads_to_add,
                                                                                        merged_basepairs_d,
                                                                                        read_id_to_basepairs_section_h,
                                                                                        read_id_to_basepairs_section_d
                                                                                       );
            device_buffer<representation_t> representations_from_this_loop_d = std::move(sketch_elements.representations_d);
            device_buffer<typename SketchElementImpl::ReadidPositionDirection> rest_from_this_loop_d = std::move(sketch_elements.rest_d);

            CGA_LOG_INFO("Deallocating {} bytes from read_id_to_basepairs_section_d", read_id_to_basepairs_section_d.size() * sizeof(decltype(read_id_to_basepairs_section_d)::value_type));
            read_id_to_basepairs_section_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from merged_basepairs_d",  merged_basepairs_d.size() * sizeof(decltype(merged_basepairs_d)::value_type));
            merged_basepairs_d.free();

            // *** sort sketch elements by representation ***
            // As this is a stable sort and the data was initailly grouper by read_id this means that the sketch elements within each representations are sorted by read_id
            thrust::stable_sort_by_key(thrust::device,
                                       representations_from_this_loop_d.data(),
                                       representations_from_this_loop_d.data() + representations_from_this_loop_d.size(),
                                       rest_from_this_loop_d.data()
                                      );

            representations_from_all_loops_h.push_back(decltype(representations_from_all_loops_h)::value_type(representations_from_this_loop_d.size()));
            CGA_CU_CHECK_ERR(cudaMemcpy(representations_from_all_loops_h.back().data(),
                                        representations_from_this_loop_d.data(),
                                        representations_from_this_loop_d.size() * sizeof(decltype(representations_from_this_loop_d)::value_type),
                                        cudaMemcpyDeviceToHost
                                       )
                            );
            rest_from_all_loops_h.push_back(typename decltype(rest_from_all_loops_h)::value_type(rest_from_this_loop_d.size()));
            CGA_CU_CHECK_ERR(cudaMemcpy(rest_from_all_loops_h.back().data(),
                                        rest_from_this_loop_d.data(),
                                        rest_from_this_loop_d.size() * sizeof(typename decltype(rest_from_this_loop_d)::value_type),
                                        cudaMemcpyDeviceToHost
                                       )
                            );

            // free these arrays as they are not needed anymore
            CGA_LOG_INFO("Deallocating {} bytes from representations_from_this_loop_d", representations_from_this_loop_d.size() * sizeof(decltype(representations_from_this_loop_d)::value_type));
            representations_from_this_loop_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from rest_from_this_loop_d", rest_from_this_loop_d.size() * sizeof(typename decltype(rest_from_this_loop_d)::value_type));
            rest_from_this_loop_d.free();

            if (parser_status == false) {
                break;
            }
        }

        // check if there is at least one sketch element (code above guarantees that each element of representations_from_all_loops_h has at least one sketch element)
        if (0 == representations_from_all_loops_h.size()){
            CGA_LOG_INFO("No Sketch Elements to be added to index");
            return;
        }

        // merge sketch elements arrays from previous arrays in one big array
        std::vector<representation_t> merged_representations_h;
        std::vector<typename SketchElementImpl::ReadidPositionDirection> merged_rest_h;

        if (representations_from_all_loops_h.size() > 1) {
            // if there is more than one array in representations_from_all_loops_h and rest_from_all_loops_h merge those arrays together
            details::index_gpu::merge_sketch_element_arrays(representations_from_all_loops_h,
                                                            rest_from_all_loops_h,
                                                            1'000'000'000, // TODO: dynamically determine this value
                                                            merged_representations_h,
                                                            merged_rest_h
                                                           );
        } else {
            // if there is only one array in each array there is nothing to be merged
            merged_representations_h = std::move(representations_from_all_loops_h[0]);
            merged_rest_h = std::move(rest_from_all_loops_h[0]);
        }

        representations_from_all_loops_h.clear();
        representations_from_all_loops_h.reserve(0);
        rest_from_all_loops_h.clear();
        rest_from_all_loops_h.reserve(0);

        /// RepresentationAndSketchElements - Representation and all sketch elements with that representation
        struct RepresentationAndSketchElements {
            /// representation
            representation_t representation_;
            /// all sketch elements with that representation (in all reads)
            std::vector<SketchElementImpl> sketch_elements_;
        };

        std::vector<SketchElementImpl> sketch_elements_for_representation;
        std::vector<RepresentationAndSketchElements> rep_to_sketch_elem;

        representation_t current_representation = merged_representations_h[0];
        // TODO: this part takes the largest portion of time
        for (std::size_t i = 0; i < merged_representations_h.size(); ++i) {
            if (merged_representations_h[i] != current_representation) {
                // New representation encountered -> add the old vector to index and start building the new one
                rep_to_sketch_elem.push_back(RepresentationAndSketchElements{current_representation, std::move(sketch_elements_for_representation)});
                sketch_elements_for_representation.clear();
                current_representation = merged_representations_h[i];
            }
            sketch_elements_for_representation.push_back(SketchElementImpl(merged_representations_h[i],
                                                                           merged_rest_h[i].position_in_read_,
                                                                           typename SketchElementImpl::DirectionOfRepresentation(merged_rest_h[i].direction_),
                                                                           merged_rest_h[i].read_id_)
                                                   );
            }
        // last representation will not be added in the loop above so add it here
        rep_to_sketch_elem.push_back(RepresentationAndSketchElements{current_representation, std::move(sketch_elements_for_representation)});

        // TODO: this is a merge of code from IndexGenerator and Index. It is going to be heavily optimized

        std::vector<std::vector<RepresentationToSketchElements>> read_id_and_representation_to_sketch_elements_temp(number_of_reads_);

        // determine the overall number of sketch elements and preallocate data arrays
        std::uint64_t total_sketch_elems = 0;
        for (const auto& sketch_elems_for_one_rep : rep_to_sketch_elem) {
            total_sketch_elems += sketch_elems_for_one_rep.sketch_elements_.size();
        }

        positions_in_reads_.reserve(total_sketch_elems);
        read_ids_.reserve(total_sketch_elems);
        directions_of_reads_.reserve(total_sketch_elems);

        // go through representations one by one
        for (const auto& sketch_elems_for_one_rep : rep_to_sketch_elem) {
            const representation_t current_rep = sketch_elems_for_one_rep.representation_;
            const auto& sketch_elems_for_current_rep = sketch_elems_for_one_rep.sketch_elements_;
            // all sketch elements with the current representation are going to be added to this section of the data arrays
            ArrayBlock array_block_for_current_rep_and_all_read_ids = ArrayBlock{positions_in_reads_.size(), static_cast<std::uint32_t>(sketch_elems_for_current_rep.size())};
            read_id_t current_read = std::numeric_limits<read_id_t>::max();
            for (const auto& sketch_elem_ptr : sketch_elems_for_current_rep) {
                const read_id_t read_of_current_sketch_elem = sketch_elem_ptr.read_id();
                // within array block for one representation sketch elements are gouped by read_id (in increasing order)
                if (read_of_current_sketch_elem != current_read) {
                    // if new read_id add new block for it
                    current_read = read_of_current_sketch_elem;
                    read_id_and_representation_to_sketch_elements_temp[read_of_current_sketch_elem].push_back(RepresentationToSketchElements{current_rep, {positions_in_reads_.size(), 1}, array_block_for_current_rep_and_all_read_ids});
                } else {
                    // if a block for that read_id alreay exists just increase the counter for the number of elements with that representation and read_id
                    ++read_id_and_representation_to_sketch_elements_temp[read_of_current_sketch_elem].back().sketch_elements_for_representation_and_read_id_.block_size_;
                }
                // add sketch element to data arrays
                positions_in_reads_.emplace_back(sketch_elem_ptr.position_in_read());
                read_ids_.emplace_back(sketch_elem_ptr.read_id());
                directions_of_reads_.emplace_back(sketch_elem_ptr.direction());
            }
        }

        std::swap(read_id_and_representation_to_sketch_elements_, read_id_and_representation_to_sketch_elements_temp);
    }

} // namespace claragenomics
