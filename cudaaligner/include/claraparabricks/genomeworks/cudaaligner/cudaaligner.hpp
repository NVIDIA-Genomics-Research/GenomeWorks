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

#include <cstdint>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{
/// \defgroup cudaaligner CUDA Aligner package
/// Base docs for the cudaaligner package (tbd)
/// \{

/// CUDA Aligner error type
enum StatusType
{
    success = 0,
    uninitialized,
    exceeded_max_alignments,
    exceeded_max_length,
    exceeded_max_alignment_difference,
    generic_error
};

/// AlignmentType - Enum for storing type of alignment.
enum AlignmentType
{
    global_alignment = 0,
    unset
};

/// AlignmentState - Enum for encoding each position in alignment.
enum AlignmentState : int8_t
{
    match = 0,
    mismatch,
    insertion, // Absent in query, present in target
    deletion   // Present in query, absent in target
};

/// Initialize CUDA Aligner context.
StatusType Init();
/// \}
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
