#pragma once

#include <memory>
#include <vector>
#include <string>

#include "cudaaligner/cudaaligner.hpp"

namespace genomeworks {

namespace cudaaligner {
/// \addtogroup cudaaligner
/// \{

/// Alignment - Object encapsulating an alignment between 2 string.
class Alignment {
    public:
        /// \brief Returns query sequence
        virtual std::string get_query_sequence() const = 0;

        /// \brief Returns target sequence
        virtual std::string get_target_sequence() const = 0;

        /// \brief Converts an alignment to CIGAR format
        ///
        /// \return CIGAR string
        virtual std::string convert_to_cigar() const = 0;

        /// \brief Returns type of alignment
        ///
        /// \return Type of alignment
        virtual AlignmentType get_alignment_type() const = 0;

        /// \brief Return status of alignment
        ///
        /// \return Status of alignment
        virtual StatusType get_status() const = 0;

        /// \brief Get the alignment between sequences
        ///
        /// \return Vector of AlignmentState encoding sequence of match,
        ///         mistmatch and insertions in alignment.
        virtual const std::vector<AlignmentState>& get_alignment() const = 0;

        /// \brief Print formatted alignment to stderr.
        virtual void print_alignment() const = 0;
};

/// \}
}

}
