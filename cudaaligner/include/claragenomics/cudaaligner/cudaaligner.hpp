/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <cstdint>

/// \defgroup cudaaligner CUDA Aligner package
/// Base docs for the cudaaligner package (tbd)
/// \{

namespace claragenomics
{

namespace cudaaligner
{
/// CUDA Aligner error type
enum class StatusType
{
    success = 0,
    uninitialized,
    exceeded_max_alignments,
    exceeded_max_length,
    exceeded_max_alignment_difference,
    generic_error
};

/// AlignmentType - Enum for storing type of alignment.
enum class AlignmentType
{
    global = 0,
    unset
};

/// AlignmentState - Enum for encoding each position in alignment.
enum class AlignmentState : int8_t
{
    match = 0,
    mismatch,
    insertion, // Present in query, absent in subject
    deletion   // Absent in query, present in subject
};

StatusType Init();
} // namespace cudaaligner
} // namespace claragenomics
/// \}
