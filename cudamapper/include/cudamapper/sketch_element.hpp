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
#include <memory>

namespace genomeworks {
/// \addtogroup cudamapper
/// \{

    /// SketchElement - Contains integer representation, position, direction and sequence id of a kmer
    class SketchElement {
    public:
        /// \brief Is this a representation of forward or reverse compliment sequence
        enum class DirectionOfRepresentation {
            FORWARD,
            REVERSE
        };

        /// \brief returns integer representation of a kmer
        /// \return integer representation
        virtual std::uint64_t representation() const = 0;

        /// \brief returns position of the sketch in the sequence
        /// \return position of the sketch in the sequence
        virtual std::size_t position() const = 0;

        /// \brief returns representation's direction
        /// \return representation's direction
        virtual DirectionOfRepresentation direction() const = 0;

        /// \brief returns sequence's ID
        /// \return sequence's ID
        virtual std::uint64_t sequence_id() const = 0;
    };

/// \}

}
