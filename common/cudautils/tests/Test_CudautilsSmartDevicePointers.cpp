/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "gtest/gtest.h"
#include "../include/cudautils/cudautils.hpp"
#include <cstdint>

namespace claragenomics
{

namespace cudautils
{

TEST(TestCudautilsSmartDevicePointers, UniquePointer)
{
    std::size_t initial_free_memory;
    std::size_t current_free_memory;
    std::size_t total_memory;

    CGA_CU_CHECK_ERR(cudaMemGetInfo(&initial_free_memory, &total_memory));

    std::unique_ptr<std::uint64_t, void (*)(std::uint64_t*)> unq_ptr = make_unique_cuda_malloc<std::uint64_t>(1048576);

    CGA_CU_CHECK_ERR(cudaMemGetInfo(&current_free_memory, &total_memory));
    EXPECT_EQ(current_free_memory + 1048576 * sizeof(std::uint64_t), initial_free_memory);

    unq_ptr = nullptr;

    CGA_CU_CHECK_ERR(cudaMemGetInfo(&current_free_memory, &total_memory));
    EXPECT_EQ(current_free_memory, initial_free_memory);
}

TEST(TestCudautilsSmartDevicePointers, SharedPointer)
{
    std::size_t initial_free_memory;
    std::size_t current_free_memory;
    std::size_t total_memory;

    CGA_CU_CHECK_ERR(cudaMemGetInfo(&initial_free_memory, &total_memory));

    std::shared_ptr<std::uint64_t> shr_ptr_1 = make_shared_cuda_malloc<std::uint64_t>(1048576);

    CGA_CU_CHECK_ERR(cudaMemGetInfo(&current_free_memory, &total_memory));
    EXPECT_EQ(current_free_memory + 1048576 * sizeof(std::uint64_t), initial_free_memory);

    std::shared_ptr<std::uint64_t> shr_ptr_2 = shr_ptr_1;

    CGA_CU_CHECK_ERR(cudaMemGetInfo(&current_free_memory, &total_memory));
    EXPECT_EQ(current_free_memory + 1048576 * sizeof(std::uint64_t), initial_free_memory);

    shr_ptr_1 = nullptr;

    CGA_CU_CHECK_ERR(cudaMemGetInfo(&current_free_memory, &total_memory));
    EXPECT_EQ(current_free_memory + 1048576 * sizeof(std::uint64_t), initial_free_memory);

    shr_ptr_2 = nullptr;

    CGA_CU_CHECK_ERR(cudaMemGetInfo(&current_free_memory, &total_memory));
    EXPECT_EQ(current_free_memory, initial_free_memory);
}

} // namespace cudautils

} // namespace claragenomics
