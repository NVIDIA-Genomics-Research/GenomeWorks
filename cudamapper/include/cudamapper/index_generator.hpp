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

namespace genomeworks {
/// \addtogroup cudamapper
/// \{

    /// IndexGenerator - generates data structures necessary for building the actualy index
    class IndexGenerator {
        public:

        /// \brief returns a mapping of sketch element representations to all sketch elements with those representations
        /// \return mapping of sketch element representations to all sketch elements with those representations
        virtual const std::map<representation_t, std::vector<std::unique_ptr<SketchElement>>>& representation_to_sketch_elements() const = 0;

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

/*namespace genomeworks {
/// \addtogroup cudamapper
/// \{

    /// IndexGenerator - generates a hash table which maps kmer integer representations to all occurrences of that representation
    class IndexGenerator {
    public:
        /// \brief returns the hash map
        ///
        /// \return hash map
        virtual const std::unordered_multimap<std::uint64_t, std::unique_ptr<SketchElement>>& representation_sketch_element_mapping() const = 0;

        /// \brief create_index - return an IndexGenerator object
        ///
        /// Given one or more sequences generates an in-memory mapping of (k,w)-kmer-representations and all occurences of that representation
        ///
        /// \param query_filename filepath to reads in FASTA or FASTQ format
        /// \param kmer_length k - the kmer lenght
        /// \param window_size w - the length of the sliding window
        ///
        /// \return Index implementation, generates representation - sketch elements mapping
        static std::unique_ptr<IndexGenerator> create_index_generator(const std::string &query_filename, std::uint64_t kmer_length, std::uint64_t window_size);
    };

/// \}

}*/
