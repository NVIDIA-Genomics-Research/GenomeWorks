/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <cstdint>
#include <memory>
#include <claraparabricks/genomeworks/cudamapper/types.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
/// \addtogroup cudamapper
/// \{

/// SketchElement - Contains integer representation, position, direction and read id of a kmer
class SketchElement
{
public:
    /// \brief Is this a representation of forward or reverse compliment
    enum class DirectionOfRepresentation
    {
        FORWARD,
        REVERSE
    };

    /// \brief Virtual destructor for SketchElement
    virtual ~SketchElement() = default;

    /// \brief returns integer representation of a kmer
    /// \return integer representation
    virtual representation_t representation() const = 0;

    /// \brief returns position of the sketch in the read
    /// \return position of the sketch in the read
    virtual position_in_read_t position_in_read() const = 0;

    /// \brief returns representation's direction
    /// \return representation's direction
    virtual DirectionOfRepresentation direction() const = 0;

    /// \brief returns read ID
    /// \return read ID
    virtual read_id_t read_id() const = 0;
};

/// \}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
