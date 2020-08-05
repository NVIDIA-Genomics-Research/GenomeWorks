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
#include <vector>
#include <claraparabricks/genomeworks/cudamapper/sketch_element.hpp>
#include <claraparabricks/genomeworks/cudamapper/types.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

/// Minimizer - represents one occurrance of a minimizer
class Minimizer : public SketchElement
{
public:
    /// \brief constructor
    ///
    /// \param representation 2-bit packed representation of a kmer
    /// \param position position of the minimizer in the read
    /// \param direction in which the read was read (forward or reverse complimet)
    /// \param read_id read's id
    Minimizer(representation_t representation, position_in_read_t position_in_read, DirectionOfRepresentation direction, read_id_t read_id);

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

    /// \brief read_id, position_in_read and direction of a minimizer
    struct ReadidPositionDirection
    {
        // read id
        read_id_t read_id_;
        // position in read
        position_in_read_t position_in_read_;
        // direction
        char direction_;
    };

    // TODO: this will be replaced with Minimizer
    /// \brief a collection of sketch element
    struct GeneratedSketchElements
    {
        // representations of sketch elements
        device_buffer<representation_t> representations_d;
        // read_ids, positions_in_reads and directions of sketch elements. Each element from this data structure corresponds to the element with the same index from representations_d
        device_buffer<ReadidPositionDirection> rest_d;
    };

    /// \brief generates sketch elements from the given input
    ///
    /// \param number_of_reads_to_add number of reads which should be added to the collection (= number of reads in the data that is passed to the function)
    /// \param minimizer_size
    /// \param window_size
    /// \param read_id_of_first_read read_id numbering in the output should should be offset by this value
    /// \param merged_basepairs_d basepairs of all reads, gouped by reads (device memory)
    /// \param read_id_to_basepairs_section_h for each read_id points to the section of merged_basepairs_d that belong to that read_id (host memory)
    /// \param read_id_to_basepairs_section_h for each read_id points to the section of merged_basepairs_d that belong to that read_id (device memory)
    /// \param hash_minimizers if true, apply a hash function to the representations
    /// \param cuda_stream CUDA stream on which the work is to be done
    static GeneratedSketchElements generate_sketch_elements(DefaultDeviceAllocator allocator,
                                                            const std::uint64_t number_of_reads_to_add,
                                                            const std::uint64_t minimizer_size,
                                                            const std::uint64_t window_size,
                                                            const std::uint64_t read_id_of_first_read,
                                                            const device_buffer<char>& merged_basepairs_d,
                                                            const std::vector<ArrayBlock>& read_id_to_basepairs_section_h,
                                                            const device_buffer<ArrayBlock>& read_id_to_basepairs_section_d,
                                                            const bool hash_representations = true,
                                                            const cudaStream_t cuda_stream  = 0);

private:
    representation_t representation_;
    position_in_read_t position_in_read_;
    DirectionOfRepresentation direction_;
    read_id_t read_id_;
};

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
