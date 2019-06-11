/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
/*
#include <unordered_set>
#include "cudautils/cudautils.hpp"
#include "index_gpu.hpp"

namespace genomeworks {
    IndexGPU::IndexGPU(IndexGenerator& index_generator) {
        const auto& source_index = index_generator.representation_sketch_element_mapping();

        // all representations in index
        std::unordered_set<std::uint64_t> all_representations;


        // find all representation in index
        for(const auto& elem : source_index) {
            all_representations.insert(elem.first);
        }


        // host arrays to be copied to device
        std::vector<std::uint64_t> representations_h;
        std::vector<std::uint64_t> sequences_h;
        std::vector<std::size_t> positions_h;
        std::vector<SketchElement::DirectionOfRepresentation> directions_h;

        // collect the data for device arrays

        // Take representations one by one. Add all sketch elements for that representation to the arrays
        for (const std::uint64_t& representation : all_representations) {
            auto sketch_elem_range = source_index.equal_range(representation); // all sketch elements for that representation
            std::size_t occurences_of_representation = 0;
            std::size_t representation_block_start = sequences_h.size();
            representations_h.push_back(representation);
            for (auto sketch_elem_iter = sketch_elem_range.first; sketch_elem_iter != sketch_elem_range.second; ++sketch_elem_iter) {
                sequences_h.push_back((*sketch_elem_iter).second->sequence_id());
                positions_h.push_back((*sketch_elem_iter).second->position());
                directions_h.push_back((*sketch_elem_iter).second->direction());
                ++occurences_of_representation;
                sequence_ids_.insert(sequences_h.back());
                sequence_id_to_representations_.insert(std::pair<std::uint64_t, std::uint64_t>(sequences_h.back(), representation));
            }
            representation_to_device_arrays_.emplace(std::pair<std::uint64_t, MappingToDeviceArrays>(representation, {representations_h.size()-1, representation_block_start, occurences_of_representation}));
        }

        all_representations.clear();

        // Copy the data to device arrays

        size_t* temp_ptr_d = nullptr;
        GW_CU_CHECK_ERR(cudaMalloc((void**)&temp_ptr_d, representations_h.size()*sizeof(std::uint64_t)));
        representations_d_ = std::shared_ptr<std::uint64_t>(temp_ptr_d, [](std::uint64_t* p) { GW_CU_CHECK_ERR(cudaFree(p));} );
        temp_ptr_d = nullptr;
        GW_CU_CHECK_ERR(cudaMemcpy(representations_d_.get(), &representations_h[0], representations_h.size()*sizeof(std::uint64_t), cudaMemcpyHostToDevice));

        GW_CU_CHECK_ERR(cudaMalloc((void**)&temp_ptr_d, sequences_h.size()*sizeof(std::uint64_t)));
        sequences_d_ = std::shared_ptr<std::uint64_t>(temp_ptr_d, [](std::uint64_t* p) { GW_CU_CHECK_ERR(cudaFree(p));} );
        temp_ptr_d = nullptr;
        GW_CU_CHECK_ERR(cudaMemcpy(sequences_d_.get(), &sequences_h[0], sequences_h.size()*sizeof(std::uint64_t), cudaMemcpyHostToDevice));

        GW_CU_CHECK_ERR(cudaMalloc((void**)&temp_ptr_d, positions_h.size()*sizeof(std::uint64_t)));
        positions_d_ = std::shared_ptr<std::uint64_t>(temp_ptr_d, [](std::uint64_t* p) { GW_CU_CHECK_ERR(cudaFree(p));} );
        temp_ptr_d = nullptr;
        GW_CU_CHECK_ERR(cudaMemcpy(positions_d_.get(), &positions_h[0], positions_h.size()*sizeof(std::uint64_t), cudaMemcpyHostToDevice));

        SketchElement::DirectionOfRepresentation* temp_ptr_direction_d;
        GW_CU_CHECK_ERR(cudaMalloc((void**)&temp_ptr_direction_d, directions_h.size()*sizeof(SketchElement::DirectionOfRepresentation)));
        directions_d_ = std::shared_ptr<SketchElement::DirectionOfRepresentation>(temp_ptr_direction_d, [](SketchElement::DirectionOfRepresentation* p) { GW_CU_CHECK_ERR(cudaFree(p));} );
        temp_ptr_direction_d = nullptr;
        GW_CU_CHECK_ERR(cudaMemcpy(directions_d_.get(), &directions_h[0], directions_h.size()*sizeof(SketchElement::DirectionOfRepresentation), cudaMemcpyHostToDevice));
    }

    IndexGPU::IndexGPU() {}

    const std::unordered_map<std::uint64_t, IndexGPU::MappingToDeviceArrays>& IndexGPU::representation_to_device_arrays() const {
        return representation_to_device_arrays_;
    }

    std::shared_ptr<const std::uint64_t> IndexGPU::representations_d() const {
        return representations_d_;
    }

    std::shared_ptr<const std::uint64_t> IndexGPU::sequence_ids_d() const {
        return sequences_d_;
    }

    std::shared_ptr<const std::size_t> IndexGPU::positions_d() const {
        return positions_d_;
    }

    std::shared_ptr<const SketchElement::DirectionOfRepresentation> IndexGPU::directions_d() const {
        return directions_d_;
    }

    const std::unordered_set<std::uint64_t> IndexGPU::sequence_ids() const {
        return sequence_ids_;
    }

    const std::unordered_multimap<std::uint64_t, std::uint64_t>& IndexGPU::sequence_id_to_representations() const {
        return sequence_id_to_representations_;
    }
}*/
