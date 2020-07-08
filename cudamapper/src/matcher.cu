

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
