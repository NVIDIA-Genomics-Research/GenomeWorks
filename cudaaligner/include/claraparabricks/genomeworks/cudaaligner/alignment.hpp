/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp>

#include <memory>
#include <vector>
#include <string>

namespace claraparabricks
{

namespace genomeworks
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

    /// \brief Returns if the alignment is optimal
    ///
    /// \return true if the alignment is optimal, false if it is an approximation
    virtual bool is_optimal() const = 0;

    /// \brief Return status of alignment
    ///
    /// \return Status of alignment
    virtual StatusType get_status() const = 0;

    /// \brief Get the alignment between sequences
    ///
    /// \return Vector of AlignmentState encoding sequence of match,
    ///         mistmatch and insertions in alignment.
    virtual const std::vector<AlignmentState>& get_alignment() const = 0;

    /// \brief Get the edit distance corrsponding to the alignment
    ///
    /// Returns the number of edits of the found alignment.
    /// If is_optimal() returns true, this is the edit distance of the two sequences.
    /// Otherwise, this number is an upper bound of the (optimal) edit distance of the two sequences.
    /// \return the number of edits of the found alignment
    virtual int32_t get_edit_distance() const = 0;

    /// \brief Print formatted alignment to stderr.
    virtual FormattedAlignment format_alignment(int32_t maximal_line_length = 80) const = 0;
};

/// \}
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
