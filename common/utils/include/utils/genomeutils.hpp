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

#include <random>

namespace genomeworks
{

inline std::string generate_random_genome(const int32_t length, std::minstd_rand& rng)
{
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    std::uniform_int_distribution<int32_t> random_index(0, 3);
    std::string genome = "";
    for (int32_t i = 0; i < length; i++)
    {
        genome += alphabet[random_index(rng)];
    }
    return genome;
}

} // namespace genomeworks
