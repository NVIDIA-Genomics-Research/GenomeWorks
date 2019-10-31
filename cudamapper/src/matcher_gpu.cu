/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "matcher_gpu.cuh"

namespace claragenomics
{

namespace cudamapper
{

MatcherGPU::MatcherGPU(const Index& query_index,
                       const Index& target_index)
{
}

std::vector<Anchor>& MatcherGPU::anchors()
{
    return anchors_h_;
}

namespace details
{

namespace matcher_gpu
{

__global__ void create_new_value_mask(const representation_t* const representations_d,
                                      const std::size_t number_of_elements,
                                      std::uint8_t* const new_value_mask_d)
{
    std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= number_of_elements)
        return;

    if (index == 0)
    {
        new_value_mask_d[0] = 1;
    }
    else
    {
        if (representations_d[index] == representations_d[index - 1])
        {
            new_value_mask_d[index] = 0;
        }
        else
            new_value_mask_d[index] = 1;
    }
}

__global__ void copy_index_of_first_occurence(const std::uint64_t* const representation_index_mask_d,
                                              const std::size_t number_of_input_elements,
                                              std::size_t* const starting_index_of_each_representation)
{
    std::uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= number_of_input_elements)
        return;

    if (index == 0)
    {
        starting_index_of_each_representation[0] = 0;
    }
    else
    {
        if (representation_index_mask_d[index] != representation_index_mask_d[index - 1])
        {
            // if new representation (= not the same as its left neighbor)
            // save the index at which that representation starts
            // representation_index_mask_d gives a unique index to each representation, starting from 1, thus '-1'
            starting_index_of_each_representation[representation_index_mask_d[index] - 1] = index;
        }
    }
}

} // namespace matcher_gpu

} // namespace details
} // namespace cudamapper

} // namespace claragenomics
