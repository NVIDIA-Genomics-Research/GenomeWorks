/*
* Copyright 2020 NVIDIA CORPORATION.
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

#include <claraparabricks/genomeworks/cudasequence/alphabet.hpp>
#include <memory>
#include <string_view>

namespace claraparabricks
{

namespace genomeworks
{

class SequenceVector
{
public:
    SequenceVector() = default;
    SequenceVector(std::shared_ptr<Alphabet> alphabet, int64_t max_bases_total);
    bool push_back_string(std::string_view x, bool is_reverse_complement);

    SequenceView operator[](int64_t index) const;

private:
    std::shared_ptr<Alphabet> alphabet_;
};

std::shared_ptr<Alphabet> make_alphabet(std::string_view letters);

} // namespace genomeworks
} // namespace claraparabricks
