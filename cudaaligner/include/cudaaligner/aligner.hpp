#pragma once

#include <memory>

namespace genomeworks {

namespace cudaaligner {

// Forward declaration of Alignment class.
class Alignment;

/// \addtogroup cudaaligner
/// \{

/// \class Aligner
/// CUDA Alignment object
class Aligner {
    public:

        /// \brief Perform CUDA accelerated alignment
        ///
        /// Perform alignment on all Alignment objects previously
        /// inserted. The alignment objects are directly updated
        /// with the results.
        virtual void align_all() = 0;

        /// \brief Add new alignment object
        ///
        /// \param alignment Shared pointer to alignment object.
        virtual void add_alignment(std::shared_ptr<Alignment> alignment) = 0;
};

/// \brief Created Aligner object
///
/// \param max_query_length Maximum length of query string
/// \param max_target_length Maximum length of target string
/// \param max_alignments Maximum number of alignments to be performed
///
/// \return Unique pointer to Aligner object
std::unique_ptr<Aligner> create_aligner(uint32_t max_query_length, uint32_t max_target_length, uint32_t max_alignments);

/// \}

}

}
