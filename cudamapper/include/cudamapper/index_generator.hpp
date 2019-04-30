#pragma  once

#include <cstdint>
#include <memory>
#include <string>

namespace genomeworks {
/// \addtogroup cudamapper
/// \{

    /// IndexGenerator - generates and manages (k,w)-minimizer index for one or more sequences
    class IndexGenerator {
    public:
        /// \brief create_index - return an IndexGenerator object
        ///
        /// Given one or more sequences generates an-in memory (k,w)-minimizer index
        ///
        /// \param query_filename filepath to reads in FASTA or FASTQ format
        /// \param minimizer_size k
        /// \param window_size w
        ///
        /// \return Index implementation, generates minimizers indices
        static std::unique_ptr<IndexGenerator> create_index_generator(const std::string &query_filename, std::uint64_t minimizer_size, std::uint64_t window_size);
    };

/// \}

}
