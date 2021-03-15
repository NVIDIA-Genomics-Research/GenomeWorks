/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <claraparabricks/genomeworks/utils/cudautils.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudautils
{

std::size_t find_largest_contiguous_device_memory_section()
{
    // find the largest block of contiguous memory
    size_t free;
    size_t total;
    GW_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
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
                GW_CU_CHECK_ERR(cudaErrorMemoryAllocation);
                return 0;
            }
        }
        else
        {
            // if cudaMalloc failed because of error other than cudaErrorMemoryAllocation process the error
            GW_CU_CHECK_ERR(status);
            return 0;
        }
    }

    // this point should actually never be reached (loop either finds memory or causes an error)
    assert(false);
    GW_CU_CHECK_ERR(cudaErrorMemoryAllocation);
    return 0;
}

void print_error_and_abort(cudaError_t code, const char* const file, int line)
{
    std::string err = "GPU Error:: " + std::string(cudaGetErrorString(code));
    if (code == cudaErrorNoKernelImageForDevice)
    {
        err += " -- Is the code compiled for the correct GPU architecture?";
        int32_t device;
        cudaDeviceProp prop;
        if (cudaGetDevice(&device) == cudaSuccess)
        {
            if (cudaGetDeviceProperties(&prop, device) == cudaSuccess)
            {
                err += " Device has compute capability ";
                err += std::to_string(prop.major);
                err += std::to_string(prop.minor);
                err += ".";
            }
        }
    }
    err += " " + std::string(file) + " " + std::to_string(line);
    GW_LOG_ERROR(err.c_str());
    // In Debug mode, this assert will cause a debugger trap
    // which is beneficial when debugging errors.
    assert(false);
    std::abort();
}
} // namespace cudautils

} // namespace genomeworks

} // namespace claraparabricks
