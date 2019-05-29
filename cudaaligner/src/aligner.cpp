/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "cudaaligner/aligner.hpp"
#include "aligner_global.hpp"

namespace genomeworks
{

namespace cudaaligner
{

std::unique_ptr<Aligner> create_aligner(uint32_t max_query_length, uint32_t max_subject_length, uint32_t max_alignments, AlignmentType type, uint32_t device_id)
{
    if (type == AlignmentType::global)
    {
        return std::make_unique<AlignerGlobal>(max_query_length, max_subject_length, max_alignments, device_id);
    }
    else
    {
        throw std::runtime_error("Aligner for specified type not implemented yet.");
    }
}
} // namespace cudaaligner
} // namespace genomeworks
