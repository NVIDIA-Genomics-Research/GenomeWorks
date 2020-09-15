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

#include <memory>
#include <string_view>

namespace claraparabricks
{

namespace genomeworks
{

class Alphabet
{
public:
    virtual ~Alphabet()                            = default;
    virtual char encode_base(char b) const         = 0;
    virtual char decode_base(char b) const         = 0;
    virtual int32_t required_bits_per_base() const = 0;
    // TODO maybe use std::string_view?
    virtual int64_t encode_sequence(const char* input, int64_t input_size, char* output_buffer, int64_t buffer_size) const = 0;
    virtual int64_t decode_sequence(const char* input, int64_t input_size, char* output_buffer, int64_t buffer_size) const = 0;
};

std::shared_ptr<Alphabet> make_alphabet(std::string_view letters);

} // namespace genomeworks
} // namespace claraparabricks
