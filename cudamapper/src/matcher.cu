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

#include <claraparabricks/genomeworks/cudamapper/matcher.hpp>
#include "matcher_gpu.cuh"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

std::unique_ptr<Matcher> Matcher::create_matcher(DefaultDeviceAllocator allocator,
                                                 const Index& query_index,
                                                 const Index& target_index,
                                                 const cudaStream_t cuda_stream)
{
    return std::make_unique<MatcherGPU>(allocator,
                                        query_index,
                                        target_index,
                                        cuda_stream);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
