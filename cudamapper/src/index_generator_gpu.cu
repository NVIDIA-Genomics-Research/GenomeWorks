/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <algorithm>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <claragenomics/logging/logging.hpp>
#include <claragenomics/utils/cudautils.hpp>
#include <claragenomics/utils/device_buffer.cuh>

#include "bioparser/bioparser.hpp"
#include "bioparser_sequence.hpp"
#include "index_generator_gpu.hpp"
#include "cudamapper/types.hpp"
#include "cudamapper_utils.hpp"

/////////////
// TODO: this will be removed once IndexGeneratorGPU is templated
#include "minimizer.hpp"
/////////////

namespace claragenomics {

    IndexGeneratorGPU::IndexGeneratorGPU(const std::string& query_filename, std::uint64_t minimizer_size, std::uint64_t window_size)
    : minimizer_size_(minimizer_size), window_size_(window_size), index_()
    {
        generate_index(query_filename);
    }

    std::uint64_t IndexGeneratorGPU::minimizer_size() const { return minimizer_size_; }

    std::uint64_t IndexGeneratorGPU::window_size() const { return window_size_; }

    const std::vector<IndexGenerator::RepresentationAndSketchElements>& IndexGeneratorGPU::representations_and_sketch_elements() const { return index_; };

    const std::vector<std::string>& IndexGeneratorGPU::read_id_to_read_name() const { return read_id_to_read_name_; };

    const std::vector<std::uint32_t>& IndexGeneratorGPU::read_id_to_read_length() const { return read_id_to_read_length_; };

    std::uint64_t IndexGeneratorGPU::number_of_reads() const { return number_of_reads_; }

    void IndexGeneratorGPU::generate_index(const std::string &query_filename) {

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

        std::vector<std::vector<std::pair<representation_t, Minimizer::ReadidPositionDirection>>> all_representation_readid_position_direction;

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

            auto res = Minimizer::generate_sketch_elements(number_of_reads_to_add,
                                                           minimizer_size_,
                                                           window_size_,
                                                           number_of_reads_ - number_of_reads_to_add,
                                                           merged_basepairs_d,
                                                           read_id_to_basepairs_section_h,
                                                           read_id_to_basepairs_section_d
                                                          );
            auto representations_compressed_d = std::move(res.representations_d);
            auto rest_compressed_d = std::move(res.rest_d);

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
            std::vector<Minimizer::ReadidPositionDirection> rest_compressed_h(representations_compressed_d.size());
            CGA_CU_CHECK_ERR(cudaMemcpy(representations_compressed_h.data(),
                                        representations_compressed_d.data(),
                                        representations_compressed_h.size() * sizeof(decltype(representations_compressed_h)::value_type),
                                        cudaMemcpyDeviceToHost
                                       )
                            );
            CGA_CU_CHECK_ERR(cudaMemcpy(rest_compressed_h.data(),
                                        rest_compressed_d.data(),
                                        rest_compressed_h.size() * sizeof(decltype(rest_compressed_h)::value_type),
                                        cudaMemcpyDeviceToHost
                                       )
                            );

            // free these arrays as they are not needed anymore
            CGA_LOG_INFO("Deallocating {} bytes from representations_compressed_d", representations_compressed_d.size() * sizeof(decltype(representations_compressed_d)::value_type));
            representations_compressed_d.free();
            CGA_LOG_INFO("Deallocating {} bytes from rest_compressed_d", rest_compressed_d.size() * sizeof(decltype(rest_compressed_d)::value_type));
            rest_compressed_d.free();

            // now create the new one:
            std::vector<std::pair<representation_t, Minimizer::ReadidPositionDirection>> representation_read_id_position_direction;
            for(size_t i=0; i< representations_compressed_h.size(); i++){
                std::pair<representation_t, Minimizer::ReadidPositionDirection> rep_rest_pair;
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
        std::vector<std::pair<representation_t, Minimizer::ReadidPositionDirection>> repr_rest_pairs;

        merge_n_sorted_vectors(all_representation_readid_position_direction,
                               repr_rest_pairs,
                               [](const std::pair<representation_t, Minimizer::ReadidPositionDirection> &a, const std::pair<representation_t, Minimizer::ReadidPositionDirection> &b){
                                   return a.first < b.first;
                               }
                              );

        std::vector<std::unique_ptr<SketchElement>> minimizers_for_representation;
        if (repr_rest_pairs.size() < 1){
            CGA_LOG_INFO("No Sketch Elements to be added to index");
            return;
        }

        representation_t current_representation = repr_rest_pairs[0].first;
        // TODO: this part takes the largest portion of time
        for (std::size_t i = 0; i < repr_rest_pairs.size(); ++i) {
            if (repr_rest_pairs[i].first != current_representation) {
                // New representation encountered -> add the old vector to index and start building the new one
                index_.push_back(RepresentationAndSketchElements{current_representation, std::move(minimizers_for_representation)});
                minimizers_for_representation.clear();
                current_representation = repr_rest_pairs[i].first;
            }
            minimizers_for_representation.push_back(std::make_unique<Minimizer>(repr_rest_pairs[i].first,
                                                                                repr_rest_pairs[i].second.position_in_read_,
                                                                                SketchElement::DirectionOfRepresentation(repr_rest_pairs[i].second.direction_),
                                                                                repr_rest_pairs[i].second.read_id_));
            }
        // last representation will not be added in the loop above so add it here
        index_.push_back(RepresentationAndSketchElements{current_representation, std::move(minimizers_for_representation)});
    }
}
