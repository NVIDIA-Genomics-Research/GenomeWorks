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

#include <map>
#include <memory>
#include <vector>
#include "cudamapper/index.hpp"
#include "cudamapper/index_generator.hpp"
#include "cudamapper/sketch_element.hpp"
#include "cudamapper/types.hpp"

namespace claragenomics {

    class IndexCPU : public Index {
    public:

        IndexCPU(const IndexGenerator& index_generator);

        IndexCPU();

        std::shared_ptr<position_in_read_t> positions_in_read_h() const;

        std::shared_ptr<representation_t> representations_h() const;

        std::shared_ptr<SketchElement::DirectionOfRepresentation> direction_of_reads_h() const;

        std::uint64_t number_of_reads() const;

        const std::vector<std::map<representation_t, ArrayBlock>> array_block_for_representations_and_reads() const;

        const std::map<representation_t, ArrayBlock> representation_to_all_its_sketch_elements() const;

    private:

    };
        

}
