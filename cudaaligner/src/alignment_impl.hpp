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

#include <claraparabricks/genomeworks/cudaaligner/alignment.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

class AlignmentImpl : public Alignment
{
public:
    AlignmentImpl(const char* query, int32_t query_length, const char* target, int32_t target_length);

    /// \brief Returns query sequence
    const std::string& get_query_sequence() const override
    {
        return query_;
    }

    /// \brief Returns target sequence
    const std::string& get_target_sequence() const override
    {
        return target_;
    }

    /// \brief Converts an alignment to CIGAR format
    ///
    /// \return CIGAR string
    std::string convert_to_cigar() const override;

    /// \brief Set alignment type.
    /// \param type Alignment type.
    virtual void set_alignment_type(AlignmentType type)
    {
        type_ = type;
    }

    /// \brief Returns type of alignment
    ///
    /// \return Type of alignment
    AlignmentType get_alignment_type() const override
    {
        return type_;
    }

    /// \brief Returns if the alignment is optimal
    ///
    /// \return true if the alignment is optimal, false if it is an approximation
    bool is_optimal() const override
    {
        return is_optimal_;
    }

    /// \brief Set status of alignment
    /// \param status Status to set for the alignment
    virtual void set_status(StatusType status)
    {
        status_ = status;
    }

    /// \brief Return status of alignment
    ///
    /// \return Status of alignment
    StatusType get_status() const override
    {
        return status_;
    }

    /// \brief Set alignment between sequences.
    ///
    /// \param alignment Alignment between sequences
    /// \param is_optimal true if the alignment is optimal, false if it is an approximation
    virtual void set_alignment(const std::vector<AlignmentState>& alignment, bool is_optimal)
    {
        alignment_  = alignment;
        is_optimal_ = is_optimal;
    }

    /// \brief Get the alignment between sequences
    ///
    /// \return Vector of AlignmentState encoding sequence of match,
    ///         mistmatch and insertions in alignment.
    const std::vector<AlignmentState>& get_alignment() const override
    {
        return alignment_;
    }

    /// \brief Get the edit distance corrsponding to the alignment
    ///
    /// Returns the number of edits of the found alignment.
    /// If is_optimal() returns true, this is the edit distance of the two sequences.
    /// Otherwise, this number is an upper bound of the (optimal) edit distance of the two sequences.
    /// \return the number of edits of the found alignment
    int32_t get_edit_distance() const override;

    /// \brief Print formatted alignment to stderr.
    FormattedAlignment format_alignment(int32_t maximal_line_length = 80) const override;

private:
    std::string query_;
    std::string target_;
    StatusType status_;
    AlignmentType type_;
    std::vector<AlignmentState> alignment_;
    bool is_optimal_;
};
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
