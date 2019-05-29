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

#include <cstdlib>

namespace genomeworks
{

namespace cudaaligner
{

inline std::string generate_random_genome(const uint32_t length)
{
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    std::string genome     = "";
    for (uint32_t i = 0; i < length; i++)
    {
        genome += alphabet[rand() % 4];
    }
    return genome;
}

} // namespace cudaaligner

} // namespace genomeworks
