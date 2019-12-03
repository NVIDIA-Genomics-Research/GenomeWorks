/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claragenomics/cudaaligner/alignment.hpp>

#include <iostream>

namespace claragenomics
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
} // namespace claragenomics
