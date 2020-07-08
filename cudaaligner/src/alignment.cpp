

#include <claraparabricks/genomeworks/cudaaligner/alignment.hpp>

#include <iostream>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

std::ostream& operator<<(std::ostream& os, const FormattedAlignment& formatted_alignment)
{
    std::size_t line_length = (formatted_alignment.linebreak_after == 0) ? formatted_alignment.query.size() : formatted_alignment.linebreak_after;
    for (std::size_t i = 0; i < formatted_alignment.query.size(); i += line_length)
    {
        os << formatted_alignment.query.substr(i, line_length) << '\n'
           << formatted_alignment.pairing.substr(i, line_length) << '\n'
           << formatted_alignment.target.substr(i, line_length) << '\n';
    }
    os << std::endl;
    return os;
}

} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
