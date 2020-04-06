/*
* Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <claragenomics/utils/cudautils.hpp>

namespace claragenomics
{

namespace cudautils
{

std::size_t find_largest_contiguous_device_memory_section()
{
    // find the largest block of contiguous memory
    size_t free;
    size_t total;
    CGA_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
    const size_t memory_decrement = free / 100;              // decrease requested memory one by one percent
    size_t size_to_try            = free - memory_decrement; // do not go for all memory
    while (true)
    {
        void* dummy_ptr    = nullptr;
        cudaError_t status = cudaMalloc(&dummy_ptr, size_to_try);
        // if it was able to allocate memory free the memory and return the size
        if (status == cudaSuccess)
        {
            cudaFree(dummy_ptr);
            return size_to_try;
        }

        if (status == cudaErrorMemoryAllocation)
        {
            // if it was not possible to allocate the memory because there was not enough of it
            // try allocating less memory in next iteration
            if (size_to_try > memory_decrement)
            {
                size_to_try -= memory_decrement;
            }
            else
            { // a very small amount of memory left, report an error
                CGA_CU_CHECK_ERR(cudaErrorMemoryAllocation);
                return 0;
            }
        }
        else
        {
            // if cudaMalloc failed because of error other than cudaErrorMemoryAllocation process the error
            CGA_CU_CHECK_ERR(status);
            return 0;
        }
    }

    // this point should actually never be reached (loop either finds memory or causes an error)
    assert(false);
    CGA_CU_CHECK_ERR(cudaErrorMemoryAllocation);
    return 0;
}
} // namespace cudautils
} // namespace claragenomics
