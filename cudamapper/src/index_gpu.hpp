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
#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "cudamapper/index.hpp"
#include "index_generator_cpu.hpp"

namespace genomeworks {
    /// IndexGPU - index of sketch elements
    ///
    /// Index of sketch elemetnts suitable for GPU processing
    /// Each sketch element represents one kmer and consists of representation (invertible hash of kmer itself), sequence id,
    /// position in the sequence and the direction (forward or inverse of original kmer)
    ///
    /// Index is split across host and device memory.
    /// Device memory contains the following arrays:
    ///     - representations (one element per representation)
    ///     - sequences (one element per sketch element)
    ///     - positions (one element per sketch element)
    ///     - directions (one element per sketch element)
    /// Elements of sequences, positions and directions arrays which belong to sketch elements with the same representation are grouped together
    /// Mappings of sketch elements to sequences, positions and directions arrays are the same for all three arrays
    ///
    /// Host memory contains a hash table which maps a representation to parts of device arrays which correspond to sketch elements with that representation.
    /// Hash map has the following structure:
    ///     - Key
    ///         - kmer representation
    ///     - Value
    ///         - location in representations array
    ///         - location of the first element corresponding to that representation in sequences, positions and directions arrays
    ///         - total number of elements corresponding to that representation in sequences, positions and directions arrays
    class IndexGPU : public Index {
    public:
        /// MappingToArrays - positions in device arrays relevant for a sketch element
        struct MappingToDeviceArrays {
            /// Location where representation value is saved
            std::size_t location_representation_;
            /// First element in sequences, positions and directions arrays corresponding to that representation
            std::size_t location_first_in_block_;
            /// total number of elements with that representation
            std::size_t block_size_;
        };

        /// \brief Constructs the index using index_generator
        ///
        /// \param index_generator
        IndexGPU(IndexGenerator& index_generator);

        /// \brief Constructs an emptry index
        IndexGPU();

        /// \brief Returns a hash table which maps representation to parts of corresponding device arrays
        ///
        /// \return hash table which maps representation to parts of corresponding device arrays
        const std::unordered_map<std::uint64_t, MappingToDeviceArrays>& representation_to_device_arrays() const;

        /// \brief Returns and array of representations (device memory).
        ///
        /// Use representation_to_device_arrays() to find the element which corresponds to a given representation
        ///
        /// \return array of representations (device memory)
        std::shared_ptr<const std::uint64_t> representations_d() const;

        /// \brief Returns an array of sequence ids (device memory)
        ///
        /// Sequence ids that belong to sketch elements with the same representation are grouped together
        /// Use representation_to_device_arrays() to find the range of elements which corresponds to a given representation
        ///
        /// \return array of squence ids (device memory)
        std::shared_ptr<const std::uint64_t> sequence_ids_d() const;

        /// \brief Returns an array of positions of sketch elements in its sequence (device memory)
        ///
        /// Positions that belong to sketch elements with the same representation are grouped together
        /// Use representation_to_device_arrays() to find the range of elements which corresponds to a given representation
        ///
        /// \return array of positions (device memory)
        std::shared_ptr<const std::size_t> positions_d() const;

        /// \brief Returns an array of directions (device memory)
        ///
        /// Directions that belong to sketch elements with the same representation are grouped together
        /// Use representation_to_device_arrays() to find the range of elements which corresponds to a given representation
        ///
        /// \return array of directions (device memory)
        std::shared_ptr<const SketchElement::DirectionOfRepresentation> directions_d() const;

        /// Returns all sequence ids
        ///
        /// \return set with all sequence ids
        const std::unordered_set<std::uint64_t> sequence_ids() const;

        /// \brief Maps sequence id to all representations in that sequence
        ///
        /// \return mapping of sequence id to all representations in that sequence
        const std::unordered_multimap<std::uint64_t, std::uint64_t>& sequence_id_to_representations() const;
    private:
        /// hash table that maps representations to relevant parts of device arrays
        std::unordered_map<std::uint64_t, MappingToDeviceArrays> representation_to_device_arrays_;

        /// representation values
        std::shared_ptr<std::uint64_t> representations_d_;

        /// sequence ids of sketch elements
        std::shared_ptr<std::uint64_t> sequences_d_;

        /// positions of sketch elements in the sequence
        std::shared_ptr<std::size_t> positions_d_;

        /// directions of sketch elements
        std::shared_ptr<SketchElement::DirectionOfRepresentation> directions_d_;

        /// all sequence ids
        std::unordered_set<std::uint64_t> sequence_ids_;

        /// maps sequence id to all representations in that sequence
        std::unordered_multimap<std::uint64_t, std::uint64_t> sequence_id_to_representations_;
    };
}
*/
