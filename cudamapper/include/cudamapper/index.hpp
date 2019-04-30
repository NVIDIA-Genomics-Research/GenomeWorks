#pragma once

#include <memory>
#include "cudamapper/index_generator.hpp"


namespace genomeworks {
/// \addtogroup cudamapper
/// \{

    /// Index - manages mapping of (k,w)-kmer-representation and all its occurences
    class Index {
    public:
        /// \brief generates a mapping of (k,w)-kmer-representation to all of its occurrences for one or more sequences
        ///
        /// \return index
        static std::unique_ptr<Index> create_index(IndexGenerator& index_generator);

        /// \brief creates an empty index
        ///
        /// \return empty index
        static std::unique_ptr<Index> create_index();
    };

/// \}

}
