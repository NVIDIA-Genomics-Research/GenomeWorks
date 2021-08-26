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

#ifndef GW_INCLUDED_DEVICE_ALLOCATOR_HPP
#define GW_INCLUDED_DEVICE_ALLOCATOR_HPP

#ifdef GW_ENABLE_CACHING_ALLOCATOR
#warning "GW_ENABLE_CACHING_ALLOCATOR should not be already set."
#else
#define GW_ENABLE_CACHING_ALLOCATOR
#endif

#include "allocators.hpp"

namespace claraparabricks
{

namespace genomeworks
{

using DefaultDeviceAllocator = CachingDeviceAllocator<char, details::DevicePreallocatedAllocator>;

/// Constructs a DefaultDeviceAllocator
///
/// This function provides a way to construct a valid DefaultDeviceAllocator
/// for all possible DefaultDeviceAllocators.
/// Use this function to obtain a DefaultDeviceAllocator object.
/// This function is needed, since construction of CachingDeviceAllocator
/// requires a max_caching_size argument to obtain a valid allocator.
/// Default constuction of CachingDeviceAllocator yields an dummy object
/// which cannot allocate memory.
/// \param max_cached_bytes max bytes used by memory resource used by CachingDeviceAllocator (default: 2GiB)
/// \param default_stream if a call to allocate() does not specify any streams this stream will be used instead
inline DefaultDeviceAllocator create_default_device_allocator(std::size_t max_caching_size = 2ull * 1024 * 1024 * 1024,
                                                              cudaStream_t default_stream  = 0)
{
    return DefaultDeviceAllocator(max_caching_size,
                                  default_stream);
}

} // namespace genomeworks

} // namespace claraparabricks

#else
#error "Attempted to included 2 DeviceAllocators!"
#endif
