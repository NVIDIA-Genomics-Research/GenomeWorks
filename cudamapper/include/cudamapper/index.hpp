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

#include <memory>
#include "cudamapper/index_generator.hpp"

namespace claragenomics {
/// \addtogroup cudamapper
/// \{

    /// Index - manages mapping of (k,w)-kmer-representation and all its occurences
    class Index {
    public:

        /// RepresentationToSketchElements - representation, pointer to section of data arrays with sketch elements with that representation and a given read_id, and a pointer to section of data arrays with sketch elements with that representation and all read_ids
        struct RepresentationToSketchElements {
            /// representation
            representation_t representation_;
            /// pointer to all sketch elements for that representation in some read (no need to save which one)
            ArrayBlock sketch_elements_for_representation_and_read_id_;
            /// pointer to all sketch elements with that representation in all reads
            ArrayBlock sketch_elements_for_representation_and_all_read_ids_;
        };

        /// \brief Virtual destructor for Index
        virtual ~Index() = default;

        /// \brief returns an array of starting positions of sketch elements in their reads
        /// \return an array of starting positions of sketch elements in their reads
        virtual const std::vector<position_in_read_t>& positions_in_reads() const = 0;

        /// \brief returns an array of reads ids for sketch elements
        /// \return an array of reads ids for sketch elements
        virtual const std::vector<read_id_t>& read_ids() const = 0;


        /// \brief returns an array of directions in which sketch elements were read
        /// \return an array of directions in which sketch elements were read
        virtual const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads() const = 0;

        /// \brief returns number of reads in input data
        /// \return number of reads in input data
        virtual std::uint64_t number_of_reads() const = 0;

        /// \brief returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        /// \return mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        virtual const std::vector<std::string>& read_id_to_read_name() const = 0;

        /// \brief returns mapping of internal read id that goes from 0 to read lengths for that read
        /// \return mapping of internal read id that goes from 0 to read lengths for that read
        virtual const std::vector<std::uint32_t>& read_id_to_read_length() const =0;

        /// \brief For each read_id (outer vector) returns a vector in which each element contains a representation from that read, pointer to section of data arrays with sketch elements with that representation and that read_id, and pointer to section of data arrays with skecth elements with that representation and all read_ids. There elements are sorted by representation in increasing order
        /// \return the mapping
        virtual const std::vector<std::vector<RepresentationToSketchElements>>& read_id_and_representation_to_sketch_elements() const = 0;

        /// \brief generates a mapping of (k,w)-kmer-representation to all of its occurrences for one or more sequences
        /// \return instance of Index
        static std::unique_ptr<Index> create_index(const IndexGenerator& index_generator);

        /// \brief creates an empty Index
        /// \return empty instacne of Index
        static std::unique_ptr<Index> create_index();
    };

/// \}

}
