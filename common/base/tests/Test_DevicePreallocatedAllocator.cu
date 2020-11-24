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

#include "gtest/gtest.h"

#include <vector>

#include <claraparabricks/genomeworks/types.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include <claraparabricks/genomeworks/utils/device_preallocated_allocator.cuh>

namespace claraparabricks
{

namespace genomeworks
{

TEST(TestDevicePreallocatedAllocator, allocations_do_not_overlap)
{
    CudaStream cuda_stream = make_cuda_stream();

    std::vector<cudaStream_t> cuda_streams;
    cuda_streams.push_back(cuda_stream.get());

    DevicePreallocatedAllocator allocator(2000);
    // 0 - 1999: free

    cudaError status;

    void* pointer_from_0_to_999_actually_to_1023 = nullptr;
    status                                       = allocator.DeviceAllocate(&pointer_from_0_to_999_actually_to_1023, 1000, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1023: taken
    // 1024 - 1999: free

    void* pointer_from_1024_to_1523_actually_to_1535 = nullptr;
    status                                           = allocator.DeviceAllocate(&pointer_from_1024_to_1523_actually_to_1535, 500, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1535: taken
    // 1536 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1024_to_1523_actually_to_1535) - static_cast<gw_byte_t*>(pointer_from_0_to_999_actually_to_1023), 1024);

    void* pointer_from_1536_to_1537_actually_to_1791 = nullptr;
    status                                           = allocator.DeviceAllocate(&pointer_from_1536_to_1537_actually_to_1791, 2, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1791: taken
    // 1792 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1536_to_1537_actually_to_1791) - static_cast<gw_byte_t*>(pointer_from_1024_to_1523_actually_to_1535), 512);

    void* pointer_from_1792_to_1999_actually_to_1999 = nullptr;
    status                                           = allocator.DeviceAllocate(&pointer_from_1792_to_1999_actually_to_1999, 208, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1999: taken
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1792_to_1999_actually_to_1999) - static_cast<gw_byte_t*>(pointer_from_1536_to_1537_actually_to_1791), 256);

    status = allocator.DeviceFree(pointer_from_1792_to_1999_actually_to_1999);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_1536_to_1537_actually_to_1791);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_1024_to_1523_actually_to_1535);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_0_to_999_actually_to_1023);
    ASSERT_EQ(status, cudaSuccess);
}

TEST(TestDevicePreallocatedAllocator, memory_correctly_deallocated)
{
    CudaStream cuda_stream = make_cuda_stream();

    std::vector<cudaStream_t> cuda_streams;
    cuda_streams.push_back(cuda_stream.get());

    DevicePreallocatedAllocator allocator(2000);
    // 0 - 1999: free

    cudaError status;

    void* pointer_from_0_to_999_actually_to_1023 = nullptr;
    status                                       = allocator.DeviceAllocate(&pointer_from_0_to_999_actually_to_1023, 1000, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1023: taken
    // 1024 - 19999: free

    void* pointer_from_1024_to_1523_actually_to_1535 = nullptr;
    status                                           = allocator.DeviceAllocate(&pointer_from_1024_to_1523_actually_to_1535, 500, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1535: taken
    // 1536 - 19999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1024_to_1523_actually_to_1535) - static_cast<gw_byte_t*>(pointer_from_0_to_999_actually_to_1023), 1024);

    void* pointer_from_1536_to_1537_actually_to_1791 = nullptr;
    status                                           = allocator.DeviceAllocate(&pointer_from_1536_to_1537_actually_to_1791, 2, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1791: taken
    // 1792 - 19999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1536_to_1537_actually_to_1791) - static_cast<gw_byte_t*>(pointer_from_1024_to_1523_actually_to_1535), 512);

    status = allocator.DeviceFree(pointer_from_1024_to_1523_actually_to_1535);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1023: taken
    // 1024 - 1535: free
    // 1536 - 1791: taken
    // 1792 - 1999: free

    void* pointer_from_1024_to_1027_actually_1279 = nullptr;
    status                                        = allocator.DeviceAllocate(&pointer_from_1024_to_1027_actually_1279, 4, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1279: taken
    // 1280 - 1535: free
    // 1535 - 1791: taken
    // 1792 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1024_to_1027_actually_1279) - static_cast<gw_byte_t*>(pointer_from_0_to_999_actually_to_1023), 1024);

    void* pointer_from_1280_to_1300_actually_1535 = nullptr;
    status                                        = allocator.DeviceAllocate(&pointer_from_1280_to_1300_actually_1535, 21, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1791: taken
    // 1792 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1280_to_1300_actually_1535) - static_cast<gw_byte_t*>(pointer_from_1024_to_1027_actually_1279), 256);

    void* pointer_from_1792_to_1800_actually_1999 = nullptr;
    status                                        = allocator.DeviceAllocate(&pointer_from_1792_to_1800_actually_1999, 9, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1999: taken
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1792_to_1800_actually_1999) - static_cast<gw_byte_t*>(pointer_from_1536_to_1537_actually_to_1791), 256);

    status = allocator.DeviceFree(pointer_from_1280_to_1300_actually_1535);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_1024_to_1027_actually_1279);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_1536_to_1537_actually_to_1791);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_0_to_999_actually_to_1023);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1999: free

    void* pointer_from_0_to_199_actually_255 = nullptr;
    status                                   = allocator.DeviceAllocate(&pointer_from_0_to_199_actually_255, 200, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 255: taken
    // 256 - 1999: free

    void* pointer_from_256_to_260_actually_511 = nullptr;
    status                                     = allocator.DeviceAllocate(&pointer_from_256_to_260_actually_511, 5, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 511: take
    // 512 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_256_to_260_actually_511) - static_cast<gw_byte_t*>(pointer_from_0_to_199_actually_255), 256);

    void* pointer_from_512_to_515_actually_767 = nullptr;
    status                                     = allocator.DeviceAllocate(&pointer_from_512_to_515_actually_767, 4, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 767: take
    // 768 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_512_to_515_actually_767) - static_cast<gw_byte_t*>(pointer_from_256_to_260_actually_511), 256);

    status = allocator.DeviceFree(pointer_from_256_to_260_actually_511);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 255: taken
    // 256 - 511: free
    // 512 - 767: taken
    // 768 - 1999: free

    void* pointer_from_768_to_1067_actually_1279 = nullptr;
    status                                       = allocator.DeviceAllocate(&pointer_from_768_to_1067_actually_1279, 300, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 255: taken
    // 256 - 511: free
    // 512 - 1279: taken
    // 1280 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_768_to_1067_actually_1279) - static_cast<gw_byte_t*>(pointer_from_512_to_515_actually_767), 256);

    void* pointer_from_256_to_270_actually_511 = nullptr;
    status                                     = allocator.DeviceAllocate(&pointer_from_256_to_270_actually_511, 15, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1279: taken
    // 1280 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_256_to_270_actually_511) - static_cast<gw_byte_t*>(pointer_from_0_to_199_actually_255), 256);

    void* pointer_from_1280_to_1290_actually_1535 = nullptr;
    status                                        = allocator.DeviceAllocate(&pointer_from_1280_to_1290_actually_1535, 11, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    // 0 - 1535: taken
    // 1536 - 1999: free
    ASSERT_EQ(static_cast<gw_byte_t*>(pointer_from_1280_to_1290_actually_1535) - static_cast<gw_byte_t*>(pointer_from_768_to_1067_actually_1279), 512);

    status = allocator.DeviceFree(pointer_from_1280_to_1290_actually_1535);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_768_to_1067_actually_1279);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_512_to_515_actually_767);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_256_to_270_actually_511);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_0_to_199_actually_255);
    ASSERT_EQ(status, cudaSuccess);
}

TEST(TestDevicePreallocatedAllocator, no_memory_left)
{
    CudaStream cuda_stream = make_cuda_stream();

    std::vector<cudaStream_t> cuda_streams;
    cuda_streams.push_back(cuda_stream.get());

    DevicePreallocatedAllocator allocator(2000);
    // 0 - 1999: free

    cudaError status;

    void* pointer_from_0_to_1499_actually_1535 = nullptr;
    status                                     = allocator.DeviceAllocate(&pointer_from_0_to_1499_actually_1535, 1500, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    ASSERT_NE(pointer_from_0_to_1499_actually_1535, nullptr);
    // 0 - 1535: taken
    // 1536 - 1999: free

    void* pointer_from_1536_to_2000_actually_error = pointer_from_0_to_1499_actually_1535; // initially set to some value to make sure allocator.DeviceAllocate() sets it to nullptr if allocation was not successful
    status                                         = allocator.DeviceAllocate(&pointer_from_1536_to_2000_actually_error, 465, cuda_streams);
    ASSERT_EQ(status, cudaErrorMemoryAllocation);
    ASSERT_EQ(pointer_from_1536_to_2000_actually_error, nullptr);
    // 0 - 1535: taken
    // 1536 - 1999: free

    void* pointer_from_1536_to_1999_actually_1999 = nullptr;
    status                                        = allocator.DeviceAllocate(&pointer_from_1536_to_1999_actually_1999, 464, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    ASSERT_NE(pointer_from_1536_to_1999_actually_1999, nullptr);
    // 0 - 1999: taken

    status = allocator.DeviceFree(pointer_from_1536_to_1999_actually_1999);
    ASSERT_EQ(status, cudaSuccess);
    status = allocator.DeviceFree(pointer_from_0_to_1499_actually_1535);
    ASSERT_EQ(status, cudaSuccess);
}

TEST(TestDevicePreallocatedAllocator, deallocating_invalid_pointer)
{
    CudaStream cuda_stream = make_cuda_stream();

    std::vector<cudaStream_t> cuda_streams;
    cuda_streams.push_back(cuda_stream.get());

    DevicePreallocatedAllocator allocator(2000);

    cudaError status;

    void* valid_ptr = nullptr;
    status          = allocator.DeviceAllocate(&valid_ptr, 1500, cuda_streams);
    ASSERT_EQ(status, cudaSuccess);
    ASSERT_NE(valid_ptr, nullptr);

    void* invalid_ptr = static_cast<void*>(static_cast<gw_byte_t*>(valid_ptr) + 10);
    status            = allocator.DeviceFree(invalid_ptr);
    ASSERT_EQ(status, cudaErrorInvalidValue);

    void* null_ptr = nullptr;
    status         = allocator.DeviceFree(null_ptr);
    ASSERT_EQ(status, cudaSuccess);

    status = allocator.DeviceFree(valid_ptr);
    ASSERT_EQ(status, cudaSuccess);
}

} // namespace genomeworks

} // namespace claraparabricks
