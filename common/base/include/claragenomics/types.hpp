/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <cstdint>

namespace claragenomics
{

/// unique ID of a read
using read_id_t = std::uint32_t;

/// number of reads
using number_of_reads_t = read_id_t;

/// position of a basepair/kmer in a read
using position_in_read_t = std::uint32_t;

/// number of basepairs
using number_of_basepairs_t = position_in_read_t;

} // namespace claragenomics
