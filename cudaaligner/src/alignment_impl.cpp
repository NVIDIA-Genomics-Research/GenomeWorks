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

#include "alignment_impl.hpp"

#include <claraparabricks/genomeworks/utils/signed_integer_utils.hpp>
#include <algorithm>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

namespace
{

char alignment_state_to_cigar_state(AlignmentState s)
{
    // CIGAR string format from http://bioinformatics.cvr.ac.uk/blog/tag/cigar-string/
    // Implementing a reduced set of CIGAR states, covering only the M, D and I characters.
    switch (s)
    {
    case AlignmentState::match:
    case AlignmentState::mismatch: return 'M';
    case AlignmentState::insertion: return 'I';
    case AlignmentState::deletion: return 'D';
    default: throw std::runtime_error("Unrecognized alignment state.");
    }
}

} // namespace

AlignmentImpl::AlignmentImpl(const char* query, int32_t query_length, const char* target, int32_t target_length)
    : query_(query, query + throw_on_negative(query_length, "query_length has to be non-negative."))
    , target_(target, target + throw_on_negative(target_length, "target_length has to be non-negative."))
    , status_(StatusType::uninitialized)
    , type_(AlignmentType::unset)
    , alignment_()
    , is_optimal_(false)
{
    // Initialize Alignment object.
}

std::string AlignmentImpl::convert_to_cigar() const
{
    if (get_size(alignment_) < 1)
    {
        return std::string("");
    }

    std::string cigar        = "";
    char last_cigar_state    = alignment_state_to_cigar_state(alignment_[0]);
    int32_t count_last_state = 0;
    for (auto const& x : alignment_)
    {
        const char cur_cigar_state = alignment_state_to_cigar_state(x);
        if (cur_cigar_state == last_cigar_state)
        {
            count_last_state++;
        }
        else
        {
            cigar += std::to_string(count_last_state) + last_cigar_state;
            count_last_state = 1;
            last_cigar_state = cur_cigar_state;
        }
    }
    cigar += std::to_string(count_last_state) + last_cigar_state;
    return cigar;
}

int32_t AlignmentImpl::get_edit_distance() const
{
    return std::count_if(begin(alignment_), end(alignment_), [](AlignmentState s) { return s != AlignmentState::match; });
}

FormattedAlignment AlignmentImpl::format_alignment(int32_t maximal_line_length) const
{
    int64_t t_pos = 0;
    int64_t q_pos = 0;
    FormattedAlignment ret_formatted_alignment;
    ret_formatted_alignment.linebreak_after = (maximal_line_length < 0) ? 0 : maximal_line_length;

    for (auto const& x : alignment_)
    {
        switch (x)
        {
        case AlignmentState::match:
            ret_formatted_alignment.target += target_[t_pos++];
            ret_formatted_alignment.query += query_[q_pos++];
            ret_formatted_alignment.pairing += '|';
            break;
        case AlignmentState::mismatch:
            ret_formatted_alignment.target += target_[t_pos++];
            ret_formatted_alignment.query += query_[q_pos++];
            ret_formatted_alignment.pairing += 'x';
            break;
        case AlignmentState::deletion:
            ret_formatted_alignment.target += '-';
            ret_formatted_alignment.query += query_[q_pos++];
            ret_formatted_alignment.pairing += ' ';
            break;
        case AlignmentState::insertion:
            ret_formatted_alignment.target += target_[t_pos++];
            ret_formatted_alignment.query += '-';
            ret_formatted_alignment.pairing += ' ';
            break;
        default:
            throw std::runtime_error("Unknown alignment state");
        }
    }

    return ret_formatted_alignment;
}
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
