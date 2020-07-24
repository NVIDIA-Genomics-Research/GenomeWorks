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

#pragma once

#include <vector>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace claraparabricks
{

namespace genomeworks
{

/// \brief An vector using pinned host memory for fast asynchronous transfers to the GPU
///
/// It is a std::vector with a special allocator for pinned host memory.
/// Please see C++ documentation for std::vector.
/// \tparam T The object's type
template <typename T>
using pinned_host_vector = std::vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;

} // namespace genomeworks
} // namespace claraparabricks
