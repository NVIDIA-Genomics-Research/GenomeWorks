/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma  once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "cudamapper/sketch_element.hpp"
#include "cudamapper/types.hpp"

namespace claragenomics {
/// \addtogroup cudamapper
/// \{

    /// IndexGenerator - generates data structures necessary for building the actualy index
    class IndexGenerator {
        public:

        /// \brief returns a mapping of sketch element representations to all sketch elements with those representations
        /// \return mapping of sketch element representations to all sketch elements with those representations
        virtual const std::map<representation_t, std::vector<std::unique_ptr<SketchElement>>>& representations_to_sketch_elements() const = 0;

        /// \brief returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        /// returns mapping of internal read id that goes from 0 to number_of_reads-1 to actual read name from the input
        virtual const std::vector<std::string>& read_id_to_read_name() const = 0;

        virtual const std::vector<uint32_t >& read_id_to_read_length() const = 0;

        /// \brief returns number of reads
        /// \return number of reads
        virtual std::uint64_t number_of_reads() const = 0;

        /// \brief create and return an IndexGenerator object
        ///
        /// Given one or more sequences generates an in-memory mapping of (k,w)-kmer-representations and all occurences of skatch elelemtns with that representation
        ///
        /// \param query_filename filepath to reads in FASTA or FASTQ format
        /// \param kmer_length k - the kmer lenght
        /// \param window_size w - the length of the sliding window
        ///
        /// \return IndexGenerator implementation, generates representation - sketch elements mapping
        static std::unique_ptr<IndexGenerator> create_index_generator(const std::string &query_filename, std::uint64_t kmer_length, std::uint64_t window_size);
    };

/// \}
}
