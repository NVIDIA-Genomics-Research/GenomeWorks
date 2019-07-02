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
#include "minimizer.hpp"

namespace claragenomics {

    Minimizer::Minimizer(representation_t representation, position_in_read_t position_in_read, DirectionOfRepresentation direction, read_id_t read_id)
    : representation_(representation), position_in_read_(position_in_read), direction_(direction), read_id_(read_id)
    {}

    representation_t Minimizer::representation() const { return representation_; }

    position_in_read_t Minimizer::position_in_read() const { return position_in_read_; }

    read_id_t Minimizer::read_id() const { return read_id_; }

    Minimizer::DirectionOfRepresentation Minimizer::direction() const { return direction_; }

    Minimizer::RepresentationAndDirection Minimizer::kmer_to_representation(const std::string& basepairs, std::size_t start_element, std::size_t length) {
        std::uint64_t forward_representation = 0;
        std::uint64_t reverse_representation = 0;
        if (length <= 2*sizeof(representation_t)) { // two basepairs per byte due to 4-bit packing
            // TODO: Lexical ordering for now, this will change in the future
            representation_t a = 0b000;
            representation_t c = 0b001;
            representation_t g = 0b010;
            representation_t t = 0b011;
            representation_t x = 0b100;
            // in reverse complement base 0 goes to position length-1, 1 to length-2...
            for (std::size_t i = start_element, position_for_reverse = 0; i < start_element + length; ++i, ++position_for_reverse) {
                forward_representation <<= 4;
                switch(basepairs[i]) {
                    case 'A': forward_representation |= a; reverse_representation |= (t << 4*position_for_reverse); break;
                    case 'C': forward_representation |= c; reverse_representation |= (g << 4*position_for_reverse); break;
                    case 'G': forward_representation |= g; reverse_representation |= (c << 4*position_for_reverse); break;
                    case 'T': forward_representation |= t; reverse_representation |= (a << 4*position_for_reverse); break;
                    default : forward_representation |= x; reverse_representation |= (x << 4*position_for_reverse); break;
                }
            }
        } else {
            // TODO: throw?
            forward_representation = reverse_representation = std::numeric_limits<representation_t>::max();
        }

        return forward_representation <= reverse_representation ? RepresentationAndDirection{forward_representation, DirectionOfRepresentation::FORWARD} : RepresentationAndDirection{reverse_representation, DirectionOfRepresentation::REVERSE};
    }
}

