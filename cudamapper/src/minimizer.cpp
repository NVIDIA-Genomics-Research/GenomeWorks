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

        // Lookup table that takes four least significant bits of a basepair and returns four least significan bits of its reverse complement
        constexpr std::uint32_t forward_to_reverse_complement[8] = { 0b0000,
                                                                     0b0100, // A -> T (0b1 -> 0b10100)
                                                                     0b0000,
                                                                     0b0111, // C -> G (0b11 -> 0b111)
                                                                     0b0001, // T -> A (0b10100 -> 0b1)
                                                                     0b0000,
                                                                     0b0000,
                                                                     0b0011, // G -> C (0b111 -> 0b11)
                                                                  };

        if (length <= 4*sizeof(representation_t)) {
            for (std::size_t i = 0; i < length; ++i) {
                const char bp = basepairs[start_element + i];
                //forward_representation <<= 2;
                // first finds lexical ordering representation of basepair, then shifts it to the right position
                forward_representation |= (0b11 & (bp >> 2 ^ bp >> 1)) << 2*(length - 1 - i);
                reverse_representation |= (0b11 & (forward_to_reverse_complement[0b1111 & bp] >> 2 ^ forward_to_reverse_complement[0b1111 & bp] >> 1)) << 2*i;
            }
        } else {
            // TODO: throw?
            forward_representation = reverse_representation = std::numeric_limits<representation_t>::max();
        }

        return forward_representation <= reverse_representation ? RepresentationAndDirection{forward_representation, DirectionOfRepresentation::FORWARD} : RepresentationAndDirection{reverse_representation, DirectionOfRepresentation::REVERSE};
    }
}

