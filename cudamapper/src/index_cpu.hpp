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
#include <vector>
#include "cudamapper/index.hpp"
#include "cudamapper/index_generator.hpp"
#include "cudamapper/sketch_element.hpp"
#include "cudamapper/types.hpp"

namespace claragenomics {

    /// IndexCPU - Contains sketch elements grouped by representation and by read id within the representation
    ///
    /// Class contains three separate data arrays: read_ids, positions_in_reads and directions_of_reads.
    /// Elements of these three arrays with the same index represent one sketch element
    /// (read it belongs to, position in that read of the first basepair of sketch element and whether it is forward or reverse complement representation).
    /// Representation itself is not saved as it is not necessary for matching phase. It can be retrieved from the original data if needed.
    ///
    /// representation_to_all_its_sketch_elements() returns a hash map that maps a sketch element representation to section of data arrays with all sketch elements with that representation
    ///
    /// read_id_and_representation_to_all_its_sketch_elements() similarly maps read id and sketch element representation to srction of data arrays with all sketch elements with that read id representation
    ///
    /// Elements of data arrays are grouped by sketch element representation and within those groups by read id. Both representations and read ids within representations are sorted in ascending order
    class IndexCPU : public Index {
    public:

        /// \brief Constructor
        ///
        /// \param index_generator Object from to generate index
        IndexCPU(const IndexGenerator& index_generator);

        /// \brief Constructor
        IndexCPU();

        /// \brief returns an array of starting positions of sketch elements in their reads
        /// \return an array of starting positions of sketch elements in their reads
        const std::vector<position_in_read_t>& positions_in_reads() const override;

        /// \brief returns an array of reads ids for sketch elements
        /// \return an array of reads ids for sketch elements
        const std::vector<read_id_t>& read_ids() const override;

        /// \brief returns an array of directions in which sketch elements were read
        /// \return an array of directions in which sketch elements were read
        const std::vector<SketchElement::DirectionOfRepresentation>& directions_of_reads() const override;

        /// \brief returns number of reads in input data
        /// \return number of reads in input data
        std::uint64_t number_of_reads() const override;

        /// \brief returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        /// \return mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        const std::vector<std::string>& read_id_to_read_name() const override;

        const std::vector<std::uint32_t>& read_id_to_read_length() const override;

        /// \brief returns mapping of read id (vector) and representation (map) to section of data arrays with sketch elements with that read id and representation
        /// \return mapping of read id (vector) and representation (map) to section of data arrays with sketch elements with that read id and representation
        const std::vector<std::map<representation_t, ArrayBlock>>& read_id_and_representation_to_all_its_sketch_elements() const override;

        /// \brief returns mapping of representation to section of data arrays with sketch elements with that representation (and all read ids)
        /// \return mapping of representation to section of data arrays with sketch elements with that representation (and all read ids)
        const std::map<representation_t, ArrayBlock>& representation_to_all_its_sketch_elements() const override;

    private:

        std::vector<position_in_read_t> positions_in_reads_;
        std::vector<read_id_t> read_ids_;
        std::vector<SketchElement::DirectionOfRepresentation> directions_of_reads_;

        const std::uint64_t number_of_reads_;

        const std::vector<std::string> read_id_to_read_name_;

        const std::vector<std::uint32_t> read_id_to_read_length_;

        std::vector<std::map<representation_t, ArrayBlock>> read_id_and_representation_to_all_its_sketch_elements_;

        std::map<representation_t, ArrayBlock> representation_to_all_its_sketch_elements_;
    };
        

}
