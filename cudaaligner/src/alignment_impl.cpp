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

char alignment_state_to_cigar_state(const int8_t s)
{
    // CIGAR string format from http://bioinformatics.cvr.ac.uk/blog/tag/cigar-string/
    // Implementing a reduced set of CIGAR states, covering only the M, D and I characters.
    switch (s)
    {
    case AlignmentState::match:
    case AlignmentState::mismatch: return 'M';
    case AlignmentState::insertion: return 'I';
    case AlignmentState::deletion: return 'D';
    default: assert(false); return '!';
    }
}

char alignment_state_to_cigar_state_extended(const int8_t s)
{
    // CIGAR string format from http://bioinformatics.cvr.ac.uk/blog/tag/cigar-string/
    // Implementing the set of CIGAR states with =, X, D and I characters,
    // which distingishes matches and mismatches.
    switch (s)
    {
    case AlignmentState::match: return '=';
    case AlignmentState::mismatch: return 'X';
    case AlignmentState::insertion: return 'I';
    case AlignmentState::deletion: return 'D';
    default: assert(false); return '!';
    }
}

// Fills the buffer with a decimal representation of x.
//
// The buffer has to be std::numeric_limits<int32_t>::digits10 long.
// x has to be non-negative.
// It will always fill the buffer up to std::numeric_limits<int32_t>::digits10
// The number will be padded with 0s at the beginning of the buffer.
// Returns where in the array the non-zero digits start.
template <int32_t N>
int32_t append_number(char (&buffer)[N], int32_t x)
{
    constexpr int32_t max_digits = std::numeric_limits<int32_t>::digits10;
    static_assert(N == max_digits, "This function expects a buffer of size max_digits.");
    assert(x >= 0);
    std::fill_n(buffer, 0, max_digits);
    for (int32_t i = max_digits - 1; i >= 0; --i)
    {
        const int32_t y = x / 10;
        const int32_t z = x % 10;
        buffer[i]       = '0' + z;
        x               = y;
        if (x == 0)
            return i;
    }
    return 0;
}

void append_to_cigar(std::string& cigar, const int32_t runlength, const char c)
{
    constexpr int32_t max_digits = std::numeric_limits<int32_t>::digits10;
    char buffer[max_digits];
    const int32_t number_start = append_number(buffer, runlength);
    cigar.append(buffer + number_start, max_digits - number_start);
    cigar.append(1, c);
}

template <typename CigarConversionFunction>
std::string convert_to_cigar_impl(const std::vector<int8_t>& action, const std::vector<uint8_t>& runlength, CigarConversionFunction convert_to_cigar)
{
    std::string cigar;
    const int32_t length = get_size<int32_t>(action);

    if (length < 1)
    {
        return cigar;
    }
    cigar.reserve(3 * length); // I guess on average we'll get 2-digit run-lengths and 1 char for an action entry.

    char last_cigar_state    = convert_to_cigar(action[0]);
    int32_t count_last_state = runlength[0];
    for (int32_t i = 1; i < length; ++i)
    {
        const char cigar_state = convert_to_cigar(action[i]);
        if (cigar_state == last_cigar_state)
        {
            count_last_state += runlength[i];
        }
        else
        {
            append_to_cigar(cigar, count_last_state, last_cigar_state);
            last_cigar_state = cigar_state;
            count_last_state = runlength[i];
        }
    }
    append_to_cigar(cigar, count_last_state, last_cigar_state);
    return cigar;
}

template <typename CigarConversionFunction>
std::string convert_to_cigar_impl(const std::vector<AlignmentState>& alignment, CigarConversionFunction convert_to_cigar)
{
    std::string cigar;

    if (get_size(alignment) < 1)
    {
        return cigar;
    }

    char last_cigar_state    = convert_to_cigar(alignment[0]);
    int32_t count_last_state = 0;
    for (auto const& x : alignment)
    {
        const char cur_cigar_state = convert_to_cigar(x);
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

} // namespace

AlignmentImpl::AlignmentImpl(const char* const query, const int32_t query_length, const char* const target, const int32_t target_length)
    : query_(query, query + throw_on_negative(query_length, "query_length has to be non-negative."))
    , target_(target, target + throw_on_negative(target_length, "target_length has to be non-negative."))
    , status_(StatusType::uninitialized)
    , type_(AlignmentType::unset)
    , alignment_()
    , is_optimal_(false)
{
    // Initialize Alignment object.
}

std::string AlignmentImpl::convert_to_cigar(const CigarFormat format) const
{
    if (format == CigarFormat::extended)
    {
        if (!action_.empty())
        {
            return convert_to_cigar_impl(action_, runlength_, alignment_state_to_cigar_state_extended);
        }
        return convert_to_cigar_impl(alignment_, alignment_state_to_cigar_state_extended);
    }
    else
    {
        if (!action_.empty())
        {
            return convert_to_cigar_impl(action_, runlength_, alignment_state_to_cigar_state);
        }
        return convert_to_cigar_impl(alignment_, alignment_state_to_cigar_state);
    }
}

int32_t AlignmentImpl::get_edit_distance() const
{
    if (!action_.empty())
    {
        const int32_t length  = get_size<int32_t>(action_);
        int32_t edit_distance = 0;
        for (int32_t i = 0; i < length; ++i)
        {
            if (action_[i] != static_cast<int8_t>(AlignmentState::match))
            {
                edit_distance += runlength_[i];
            }
        }
        return edit_distance;
    }
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
