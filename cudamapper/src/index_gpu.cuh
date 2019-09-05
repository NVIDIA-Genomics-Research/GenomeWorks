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
        /// \param minimizer_size k - the kmer length used as a minimizer
        /// \param window_size w - the length of the sliding window used to find minimizer
        IndexGPU(const std::string& query_filename, const std::uint64_t minimizer_size, const std::uint64_t window_size);

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

        const std::uint64_t minimizer_size_;
        const std::uint64_t window_size_;
        std::uint64_t number_of_reads_;

        std::vector<position_in_read_t> positions_in_reads_;
        std::vector<read_id_t> read_ids_;
        std::vector<typename SketchElementImpl::DirectionOfRepresentation> directions_of_reads_;

        std::vector<std::string> read_id_to_read_name_;
        std::vector<std::uint32_t> read_id_to_read_length_;

        std::vector<std::vector<RepresentationToSketchElements>> read_id_and_representation_to_sketch_elements_;
    };

    template <typename SketchElementImpl>
    IndexGPU<SketchElementImpl>::IndexGPU(const std::string& query_filename, const std::uint64_t minimizer_size, const std::uint64_t window_size)
    : minimizer_size_(minimizer_size), window_size_(window_size), number_of_reads_(0)
    {
        generate_index(query_filename);
    }

    template <typename SketchElementImpl>
    IndexGPU<SketchElementImpl>::IndexGPU()
    : minimizer_size_(0), window_size_(0), number_of_reads_(0) {
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

        std::vector<std::vector<std::pair<representation_t, typename SketchElementImpl::ReadidPositionDirection>>> all_representation_readid_position_direction;

        while (true) {
            //read the query file:
            std::vector<std::unique_ptr<BioParserSequence>> fasta_objects;
            bool parser_status = query_parser->parse(fasta_objects, parser_buffer_size_in_bytes);

            std::uint64_t total_basepairs = 0;
            std::vector<ArrayBlock> read_id_to_basepairs_section_h;

            // find out how many basepairs each read has and determine its section in the big array with all basepairs
            for (std::size_t fasta_object_id = 0; fasta_object_id < fasta_objects.size(); ++fasta_object_id) {
                // skip reads which are shorter than one window
                if (fasta_objects[fasta_object_id]->data().length() >= window_size_ + minimizer_size_ - 1) {
                    read_id_to_basepairs_section_h.emplace_back(ArrayBlock{total_basepairs, static_cast<std::uint32_t>(fasta_objects[fasta_object_id]->data().length())});
                    total_basepairs += fasta_objects[fasta_object_id]->data().length();
                    read_id_to_read_name_.push_back(fasta_objects[fasta_object_id]->name());
                    read_id_to_read_length_.push_back(fasta_objects[fasta_object_id]->data().length());
                } else {
                    CGA_LOG_INFO("Skipping read {}. It has {} basepairs, one window covers {} basepairs",
                                 fasta_objects[fasta_object_id]->name(),
                                 fasta_objects[fasta_object_id]->data().length(), window_size_ + minimizer_size_ - 1
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
                if (fasta_objects[fasta_object_id]->data().length() >= window_size_ + minimizer_size_ - 1) {
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
                                                                                        minimizer_size_,
                                                                                        window_size_,
                                                                                        number_of_reads_ - number_of_reads_to_add,
                                                                                        merged_basepairs_d,
                                                                                        read_id_to_basepairs_section_h,
                                                                                        read_id_to_basepairs_section_d
                                                                                       );
            device_buffer<representation_t> representations_compressed_d = std::move(sketch_elements.representations_d);
            device_buffer<typename SketchElementImpl::ReadidPositionDirection> rest_compressed_d = std::move(sketch_elements.rest_d);

            CGA_LOG_INFO("Deallocating {} bytes from read_id_to_basepairs_section_d", read_id_to_basepairs_section_d.size() * sizeof(decltype(read_id_to_basepairs_section_d)::value_type));
            read_id_to_basepairs_section_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from merged_basepairs_d",  merged_basepairs_d.size() * sizeof(decltype(merged_basepairs_d)::value_type));
            merged_basepairs_d.free();

            // *** sort minimizers by representation ***
            // As this is a stable sort and the data was initailly grouper by read_id this means that the minimizers within each representations are sorted by read_id
            thrust::stable_sort_by_key(thrust::device, representations_compressed_d.data(),
                                       representations_compressed_d.data() + representations_compressed_d.size(),
                                       rest_compressed_d.data()
                                      );

            std::vector<representation_t> representations_compressed_h(representations_compressed_d.size());
            std::vector<typename SketchElementImpl::ReadidPositionDirection> rest_compressed_h(representations_compressed_d.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(representations_compressed_h.data(),
                                        representations_compressed_d.data(),
                                        representations_compressed_h.size() * sizeof(decltype(representations_compressed_h)::value_type),
                                        cudaMemcpyDeviceToHost
                                       )
                            );
            CGA_CU_CHECK_ERR(cudaMemcpy(rest_compressed_h.data(),
                                        rest_compressed_d.data(),
                                        rest_compressed_h.size() * sizeof(typename decltype(rest_compressed_h)::value_type),
                                        cudaMemcpyDeviceToHost
                                       )
                            );

            // free these arrays as they are not needed anymore
            CGA_LOG_INFO("Deallocating {} bytes from representations_compressed_d", representations_compressed_d.size() * sizeof(decltype(representations_compressed_d)::value_type));
            representations_compressed_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from rest_compressed_d", rest_compressed_d.size() * sizeof(decltype(rest_compressed_d)::value_type));
            rest_compressed_d.free();

            // now create the new one:
            std::vector<std::pair<representation_t, typename SketchElementImpl::ReadidPositionDirection>> representation_read_id_position_direction;
            for(size_t i=0; i< representations_compressed_h.size(); i++){
                std::pair<representation_t, typename SketchElementImpl::ReadidPositionDirection> rep_rest_pair;
                rep_rest_pair.first = representations_compressed_h[i];
                rep_rest_pair.second = rest_compressed_h[i];
                representation_read_id_position_direction.push_back(rep_rest_pair);
            }

            all_representation_readid_position_direction.push_back(representation_read_id_position_direction);

            if (parser_status == false) {
                break;
            }
        }

        // Add the minimizers to the host-side index
        // SketchElements are already sorted by representation. Add all SketchElements with the same representation to a vector and then add that vector to the index
        std::vector<std::pair<representation_t, typename SketchElementImpl::ReadidPositionDirection>> repr_rest_pairs;

        merge_n_sorted_vectors(all_representation_readid_position_direction,
                               repr_rest_pairs,
                               [](const std::pair<representation_t, typename SketchElementImpl::ReadidPositionDirection> &a, const std::pair<representation_t, typename SketchElementImpl::ReadidPositionDirection> &b){
                                   return a.first < b.first;
                               }
                              );

        std::vector<SketchElementImpl> minimizers_for_representation;
        if (repr_rest_pairs.size() < 1){
            CGA_LOG_INFO("No Sketch Elements to be added to index");
            return;
        }

        /// RepresentationAndSketchElements - Representation and all sketch elements with that representation
        struct RepresentationAndSketchElements {
            /// representation
            representation_t representation_;
            /// all sketch elements with that representation (in all reads)
            std::vector<SketchElementImpl> sketch_elements_;
        };

        std::vector<RepresentationAndSketchElements> rep_to_sketch_elem;

        representation_t current_representation = repr_rest_pairs[0].first;
        // TODO: this part takes the largest portion of time
        for (std::size_t i = 0; i < repr_rest_pairs.size(); ++i) {
            if (repr_rest_pairs[i].first != current_representation) {
                // New representation encountered -> add the old vector to index and start building the new one
                rep_to_sketch_elem.push_back(RepresentationAndSketchElements{current_representation, std::move(minimizers_for_representation)});
                minimizers_for_representation.clear();
                current_representation = repr_rest_pairs[i].first;
            }
            minimizers_for_representation.push_back(SketchElementImpl(repr_rest_pairs[i].first,
                                                                      repr_rest_pairs[i].second.position_in_read_,
                                                                      typename SketchElementImpl::DirectionOfRepresentation(repr_rest_pairs[i].second.direction_),
                                                                      repr_rest_pairs[i].second.read_id_)
                                                   );
            }
        // last representation will not be added in the loop above so add it here
        rep_to_sketch_elem.push_back(RepresentationAndSketchElements{current_representation, std::move(minimizers_for_representation)});

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
}
