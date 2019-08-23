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

#include <cstdint>
#include "cudamapper/sketch_element.hpp"
#include "cudamapper/types.hpp"

namespace claragenomics {

    /// Minimizer - represents one occurrance of a minimizer
    class Minimizer : public SketchElement {
    public:
        /// \brief constructor
        ///
        /// \param representation 2-bit packed representation of a kmer
        /// \param position position of the minimizer in the read
        /// \param direction in which the read was read (forward or reverse complimet)
        /// \param read_id read's id
        Minimizer(representation_t representation, position_in_read_t position_in_read, DirectionOfRepresentation direction, read_id_t read_id);

        /// \brief representation and its direction
        struct RepresentationAndDirection {
            representation_t representation_;
            DirectionOfRepresentation direction_;
        };

        /// \brief returns minimizers representation
        /// \return minimizer representation
        representation_t representation() const override;

        /// \brief returns position of the minimizer in the sequence
        /// \return position of the minimizer in the sequence
        position_in_read_t position_in_read() const override;

        /// \brief returns representation's direction
        /// \return representation's direction
        DirectionOfRepresentation direction() const override;

        /// \brief returns read ID
        /// \return read ID
        read_id_t read_id() const override;

        /// \brief converts a kmer of length length into 2-bit packed numeric representation
        ///
        /// Representation uses lexicographical ordering. It returns the smaller of forward and reverse complement representation
        ///
        /// \param baseparis
        /// \param start_element where in basepairs the kmer actually starts
        /// \param length length of the kmer
        ///
        /// \return representation and direction of the read
        static RepresentationAndDirection kmer_to_representation(const std::string& basepairs, std::size_t start_element, std::size_t length);

    private:
        representation_t representation_;
        position_in_read_t position_in_read_;
        DirectionOfRepresentation direction_;
        read_id_t read_id_;
    };

}

