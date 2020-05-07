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

#include <claragenomics/cudaaligner/cudaaligner.hpp>

#include <memory>
#include <vector>
#include <string>

namespace claragenomics
{

namespace cudaaligner
{
/// \addtogroup cudaaligner
/// \{

/// FormattedAlignment -Holds formatted strings representing an alignment.
typedef struct FormattedAlignment
{
    /// \brief FormattedAlignment.query = formatted string for query
    std::string query;
    /// \brief FormattedAlignment.pairing = formatted pairing string
    std::string pairing;
    /// \brief FormattedAlignment.target = formatted string for target
    std::string target;
    /// \brief Maximal line length when outputting to ostream, default = 80 (default terminal size)
    uint32_t linebreak_after = 80;
} FormattedAlignment;

/// \brief Write FormattedAlignment struct content to output stream
///
/// \return ostream
std::ostream& operator<<(std::ostream& os, const FormattedAlignment& formatted_alignment);

/// Alignment - Object encapsulating an alignment between 2 string.
class Alignment
{
public:
    /// \brief Virtual destructor
    virtual ~Alignment() = default;

    /// \brief Returns query sequence
    virtual const std::string& get_query_sequence() const = 0;

    /// \brief Returns target sequence
    virtual const std::string& get_target_sequence() const = 0;

    /// \brief Converts an alignment to CIGAR format.
    ///        The is a reduced implementation of the CIGAR standard
    ///        supporting only M, I and D states.
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
    virtual FormattedAlignment format_alignment(int32_t maximal_line_length = 80) const = 0;
};

/// \}
} // namespace cudaaligner
} // namespace claragenomics
