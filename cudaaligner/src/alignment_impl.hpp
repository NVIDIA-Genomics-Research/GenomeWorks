#pragma once

#include "cudaaligner/alignment.hpp"

namespace genomeworks {

namespace cudaaligner {

class AlignmentImpl: public Alignment
{
    public:
        AlignmentImpl(const char* query, uint32_t query_length, const char* target, uint32_t);
        ~AlignmentImpl();

        /// \brief Returns query sequence
        virtual const std::string& get_query_sequence() const override {
            return query_;
        }

        /// \brief Returns target sequence
        virtual const std::string& get_target_sequence() const override {
            return target_;
        }

        /// \brief Converts an alignment to CIGAR format
        ///
        /// \return CIGAR string
        virtual std::string convert_to_cigar() const override;

        /// \brief Set alignment type.
        /// \param type Alignment type.
        virtual void set_alignment_type(AlignmentType type) {
            type_ = type;
        }

        /// \brief Returns type of alignment
        ///
        /// \return Type of alignment
        virtual AlignmentType get_alignment_type() const override {
            return type_;
        }

        /// \brief Set status of alignment
        /// \param status Status to set for the alignment
        virtual void set_status(StatusType status) {
            status_ = status;
        }

        /// \brief Return status of alignment
        ///
        /// \return Status of alignment
        virtual StatusType get_status() const override {
            return status_;
        }
 
        /// \brief Set alignment between sequences.
        ///
        /// \param alignment Alignment between sequences
        virtual void set_alignment(const std::vector<AlignmentState>& alignment) {
            alignment_ = alignment;
        }

        /// \brief Get the alignment between sequences
        ///
        /// \return Vector of AlignmentState encoding sequence of match,
        ///         mistmatch and insertions in alignment.
        virtual const std::vector<AlignmentState>& get_alignment() const override {
            return alignment_;
        }

        /// \brief Print formatted alignment to stderr.
        virtual FormattedAlignment format_alignment() const override;
    private:

        // Query string
        std::string query_;

        // Target string
        std::string target_;

        // Status of alignment
        StatusType status_;

        // Type of alignment
        AlignmentType type_;

        // Alignment state
        std::vector<AlignmentState> alignment_;
};

}

}
