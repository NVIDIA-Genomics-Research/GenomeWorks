#pragma once

#include <memory>
#include <vector>
#include <string>

namespace genomeworks {

namespace cudaaligner {
/// \addtogroup cudaaligner
/// \{

/// AlignmentType - Enum for storing type of alignment.
enum class AlignmentType {
    GLOBAL = 0
};

/// AlignmentState - Enum for encoding each position in alignment.
enum class AlignmentState {
    MATCH = 0,
    MISMATCH,
    INSERT_INTO_QUERY,
    INSERT_INTO_TARGET
};

/// Alignment - Object encapsulating an alignment between 2 string.
class Alignment {
    public:

        /// \brief Returns query sequence
        virtual std::string get_query_sequence() = 0;

        /// \brief Returns target sequence
        virtual std::string get_target_sequence() = 0;

        /// \brief Converts an alignment to CIGAR format
        ///
        /// \return CIGAR string
        virtual std::string convert_to_cigar() = 0;

        /// \brief Returns type of alignment
        virtual AlignmentType get_alignment_type() = 0;

        /// \brief Return status of alignment
        virtual StatusType get_status() = 0;

        /// \brief Set status of alignment
        virtual void set_status() = 0;

        /// \brief Get the alignment between sequences
        ///
        /// \return Vector of AlignmentState encoding sequence of match,
        ///         mistmatch and insertions in alignment.
        virtual const std::vector<AlignmentState>& get_alignment() = 0;

        /// \brief Set alignment between sequences.
        ///
        /// \param alignment Alignment between sequences
        virtual void set_alignment(const std::vector<AlignmentState>& alignment) = 0;

        /// \brief Print formatted alignment to stderr.
        virtual void print_alignment() = 0;
};

/// \brief Construct a new Alignment object.
///
/// \param query - Query sequence
/// \param target - Target sequence
/// \param type - Type of alignment to perform
///
/// \return Shared pointer to Alignment object
std::shared_ptr<Alignment> create_alignment(const std::string& query, const std::string& target, AlignmentType type);

/// \brief Construct a new Alignment object.
///
/// \param query - Query sequence
/// \param query_length - Query length
/// \param target - Target sequence
/// \param target_length - Target length
/// \param type - Type of alignment to perform
///
/// \return Shared pointer to Alignment object
std::shared_ptr<Alignment> create_alignment(const char* query, uint32_t query_length, const char* target, uint32_t target_length, AlignmentType type);

/// \}
}

}
