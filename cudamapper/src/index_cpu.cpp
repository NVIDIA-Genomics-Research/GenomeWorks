/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <limits>
#include "index_cpu.hpp"

namespace claragenomics {

    IndexCPU::IndexCPU (const IndexGenerator& index_generator)
    : number_of_reads_(index_generator.number_of_reads()),
      read_id_to_read_name_(index_generator.read_id_to_read_name()),
      read_id_to_read_length_(index_generator.read_id_to_read_length()),
      read_id_and_representation_to_all_its_sketch_elements_(index_generator.number_of_reads()) {

        auto const& rep_to_sketch_elem = index_generator.representations_to_sketch_elements();

        // determine the overall number of sketch elements and preallocate data arrays
        std::uint64_t total_sketch_elems = 0;
        for (const auto& sketch_elems_for_one_rep : rep_to_sketch_elem) {
            total_sketch_elems += sketch_elems_for_one_rep.second.size();
        }

        positions_in_reads_.reserve(total_sketch_elems);
        read_ids_.reserve(total_sketch_elems);
        directions_of_reads_.reserve(total_sketch_elems);

        // go through representations one by one
        for (const auto& key_value_for_current_rep : rep_to_sketch_elem) {
            const representation_t current_rep = key_value_for_current_rep.first;
            const auto& sketch_elems_for_current_rep = key_value_for_current_rep.second;
            // all sketch elements with the current representation are going to be added to this section of the data arrays
            ArrayBlock array_block_for_current_rep = ArrayBlock{positions_in_reads_.size(), static_cast<std::uint32_t>(sketch_elems_for_current_rep.size())};
            representation_to_all_its_sketch_elements_.emplace(std::make_pair(current_rep, array_block_for_current_rep));
            // add all sketch elements
            read_id_t current_read = std::numeric_limits<read_id_t>::max();
            for (const auto& sketch_elem_ptr : sketch_elems_for_current_rep) {
                const read_id_t read_of_current_sketch_elem = sketch_elem_ptr->read_id();
                // within array block for one representation sketch elements are gouped by read_id (in increasing order)
                if (read_of_current_sketch_elem != current_read) {
                    // if new read_id add new block for it
                    current_read = read_of_current_sketch_elem;
                    read_id_and_representation_to_all_its_sketch_elements_[current_read].emplace(std::make_pair(current_rep, ArrayBlock{positions_in_reads_.size(), 1}));
                } else {
                    // if a block for that read_id alreay exists just increase the counter for the number of elements with that representation and read_id
                    // TODO: we're going through the map for each sketch element
                    ++read_id_and_representation_to_all_its_sketch_elements_[current_read][current_rep].block_size_;
                }
                // add sketch element to data arrays
                positions_in_reads_.emplace_back(sketch_elem_ptr->position_in_read());
                read_ids_.emplace_back(sketch_elem_ptr->read_id());
                directions_of_reads_.emplace_back(sketch_elem_ptr->direction());
            }
        }

    }

    IndexCPU::IndexCPU()
    : number_of_reads_(0) {
    }

    const std::vector<position_in_read_t>& IndexCPU::positions_in_reads() const { return positions_in_reads_; }

    const std::vector<read_id_t>& IndexCPU::read_ids() const { return read_ids_; }

    const std::vector<SketchElement::DirectionOfRepresentation>& IndexCPU::directions_of_reads() const { return directions_of_reads_; }

    std::uint64_t IndexCPU::number_of_reads() const { return number_of_reads_; }

    const std::vector<std::string>& IndexCPU::read_id_to_read_name() const { return read_id_to_read_name_; }

    const std::vector<uint32_t >& IndexCPU::read_id_to_read_length() const { return read_id_to_read_length_; }

    const std::vector<std::map<representation_t, ArrayBlock>>& IndexCPU::read_id_and_representation_to_all_its_sketch_elements() const { return read_id_and_representation_to_all_its_sketch_elements_; }

    const std::map<representation_t, ArrayBlock>& IndexCPU::representation_to_all_its_sketch_elements() const { return representation_to_all_its_sketch_elements_; }

}
