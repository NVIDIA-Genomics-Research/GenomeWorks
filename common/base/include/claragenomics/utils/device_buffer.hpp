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

#include <claragenomics/utils/buffer.hpp>
#include <claragenomics/utils/allocator.hpp>

namespace claragenomics
{

template <typename T>
#ifdef CGA_ENABLE_CACHING_ALLOCATOR
using device_buffer = buffer<T, CachingDeviceAllocator<T, DevicePreallocatedAllocator>>;
#else
using device_buffer = buffer<T, CudaMallocAllocator<T>>;
#endif

} // namespace claragenomics
