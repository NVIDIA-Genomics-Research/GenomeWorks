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

#include <vector>
#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/cudamapper/types.hpp>

namespace claragenomics
{

namespace cudamapper
{

class MatcherGPU
{
public:
    MatcherGPU(const Index& query_index,
               const Index& target_index);

    std::vector<Anchor>& anchors();

private:
    std::vector<Anchor> anchors_h_;
};

namespace details
{

namespace matcher_gpu
{

/// \brief Writes 0 to the output array if the value to the left is the same as the current value, 1 otherwise. First element is always 1
__global__ void create_new_value_mask(const representation_t* const representations_d,
                                      const std::size_t number_of_elements,
                                      std::uint8_t* const new_value_mask_d);
} // namespace matcher_gpu

} // namespace details

} // namespace cudamapper

} // namespace claragenomics
