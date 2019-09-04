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
#include "cudamapper/sketch_element.hpp"
#include "cudamapper/types.hpp"

namespace claragenomics {

    /// IndexCPU - Contains sketch elements grouped by representation and by read id within the representation
    ///
    /// Class contains three separate data arrays: read_ids, positions_in_reads and directions_of_reads.
    /// Elements of these three arrays with the same index represent one sketch element
    /// (read_id of the read it belongs to, position in that read of the first basepair of sketch element and whether it is forward or reverse complement representation).
    /// Representation itself is not saved as it is not necessary for matching phase. It can be retrieved from the original data if needed.
    ///
    /// Elements of data arrays are grouped by sketch element representation and within those groups by read_id. Both representations and read_ids within representations are sorted in ascending order
    ///
    /// read_id_and_representation_to_sketch_elements() for each read_id (outer vector) returns a vector in which each element contains a representation from that read, pointer to section of data arrays with sketch elements with that representation and that read_id, and pointer to section of data arrays with skecth elements with that representation and all read_ids. There elements are sorted by representation in increasing order
    class IndexCPU : public Index {
    public:

        /// \brief Constructor
        ///
        /// \param query_filename filepath to reads in FASTA or FASTQ format
        /// \param minimizer_size k - the kmer length used as a minimizer
        /// \param window_size w - the length of the sliding window used to find minimizer
        IndexCPU(const std::string& query_filename, const std::uint64_t minimizer_size, const std::uint64_t window_size);

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

        /// \brief returns mapping of internal read id that goes from 0 to read lengths for that read
        /// \return mapping of internal read id that goes from 0 to read lengths for that read
        const std::vector<std::uint32_t>& read_id_to_read_length() const override;

        /// \brief For each read_id (outer vector) returns a vector in which each element contains a representation from that read, pointer to section of data arrays with sketch elements with that representation and that read_id, and pointer to section of data arrays with skecth elements with that representation and all read_ids. There elements are sorted by representation in increasing order
        /// \return the mapping
        const std::vector<std::vector<Index::RepresentationToSketchElements>>& read_id_and_representation_to_sketch_elements() const override;

    private:

        /// \brief generates the index
        /// \param query_filename
        void generate_index(const std::string& query_filename);

        const std::uint64_t minimizer_size_;
        const std::uint64_t window_size_;
        std::uint64_t number_of_reads_;

        std::vector<position_in_read_t> positions_in_reads_;
        std::vector<read_id_t> read_ids_;
        std::vector<SketchElement::DirectionOfRepresentation> directions_of_reads_;

        std::vector<std::string> read_id_to_read_name_;
        std::vector<std::uint32_t> read_id_to_read_length_;

        std::vector<std::vector<RepresentationToSketchElements>> read_id_and_representation_to_sketch_elements_;
    };
}
