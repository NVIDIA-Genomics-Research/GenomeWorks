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

        /// \brief returns an array of starting positions of sketch elements in their reads
        /// \return an array of starting positions of sketch elements in their reads
        virtual const std::vector<position_in_read_t>& positions_in_reads() const = 0;

        /// \brief returns an array of reads ids for sketch elements
        /// \return an array of reads ids for sketch elements
        virtual const std::vector<read_id_t>& read_ids() const = 0;

        /// \brief returns an array of directions in which sketch elements were read
        /// \return an array of directions in which sketch elements were read
        virtual const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads() const = 0;

        /// \brief returns an array of directions in which sketch elements were read
        /// \return an array of directions in which sketch elements were read
        virtual const std::vector<std::uint32_t>& read_id_to_read_length() const =0;

        /// \brief returns number of reads in input data
        /// \return number of reads in input data
        virtual std::uint64_t number_of_reads() const = 0;

        /// \brief returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        /// \return mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        virtual const std::vector<std::string>& read_id_to_read_name() const = 0;

        /// \brief returns mapping of read id (vector) and representation (map) to section of data arrays with sketch elements with that read id and representation
        /// \return mapping of read id (vector) and representation (map) to section of data arrays with sketch elements with that read id and representation
        virtual const std::vector<std::map<representation_t, ArrayBlock>>& read_id_and_representation_to_all_its_sketch_elements() const = 0;

        /// \brief returns mapping of representation to section of data arrays with sketch elements with that representation (and all read ids)
        /// \return mapping of representation to section of data arrays with sketch elements with that representation (and all read ids)
        virtual const std::map<representation_t, ArrayBlock>& representation_to_all_its_sketch_elements() const = 0;

        /// \brief generates a mapping of (k,w)-kmer-representation to all of its occurrences for one or more sequences
        ///
        /// \return index
        static std::unique_ptr<Index> create_index(const IndexGenerator& index_generator);

        /// \brief creates an empty index
        ///
        /// \return empty index
        static std::unique_ptr<Index> create_index();
    };

/// \}

}
