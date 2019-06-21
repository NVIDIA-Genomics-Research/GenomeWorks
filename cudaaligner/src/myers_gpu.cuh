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

#include <cuda_runtime_api.h>

namespace claragenomics
{
namespace cudaaligner
{

int32_t myers_compute_edit_distance(std::string const& target, std::string const& query);

} // end namespace cudaaligner
} // end namespace claragenomics
