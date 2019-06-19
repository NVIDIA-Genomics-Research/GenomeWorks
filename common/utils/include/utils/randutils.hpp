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

#include <stdlib.h> /* srand, rand */

namespace genomeworks
{

namespace randutils
{

inline void set_rand_seed(int seed)
{
    srand(seed);
}

//returns a random integer in the range [0, range-1]
inline int rand_int_within(int range)
{
    return rand() % range;
}
//returns a random float in the range [0.0, 1.0]
inline double rand_prob()
{
    return ((double)rand() / RAND_MAX);
}

} // namespace randutils

} // namespace genomeworks
