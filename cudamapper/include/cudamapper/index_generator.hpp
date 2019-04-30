#pragma  once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include "cudamapper/sketch_element.hpp"

namespace genomeworks {
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

}
