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
