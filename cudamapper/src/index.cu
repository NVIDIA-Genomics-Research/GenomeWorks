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

#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/utils/cudautils.hpp>
#include "index_gpu.cuh"
#include "minimizer.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

std::unique_ptr<Index> Index::create_index(DefaultDeviceAllocator allocator,
                                           const io::FastaParser& parser,
                                           const read_id_t first_read_id,
                                           const read_id_t past_the_last_read_id,
                                           const std::uint64_t kmer_size,
                                           const std::uint64_t window_size,
                                           const bool hash_representations,
                                           const double filtering_parameter,
                                           const cudaStream_t cuda_stream)
{
    GW_NVTX_RANGE(profiler, "create_index");
    return std::make_unique<IndexGPU<Minimizer>>(allocator,
                                                 parser,
                                                 first_read_id,
                                                 past_the_last_read_id,
                                                 kmer_size,
                                                 window_size,
                                                 hash_representations,
                                                 filtering_parameter,
                                                 cuda_stream);
}

std::unique_ptr<IndexHostCopyBase> IndexHostCopyBase::create_cache(const Index& index,
                                                                   const read_id_t first_read_id,
                                                                   const std::uint64_t kmer_size,
                                                                   const std::uint64_t window_size,
                                                                   const cudaStream_t cuda_stream)
{
    GW_NVTX_RANGE(profiler, "cache_D2H");
    return std::make_unique<IndexHostCopy>(index,
                                           first_read_id,
                                           kmer_size,
                                           window_size,
                                           cuda_stream);
}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
