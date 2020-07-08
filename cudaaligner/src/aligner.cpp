

#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>

#include "aligner_global_hirschberg_myers.hpp"

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

std::unique_ptr<Aligner> create_aligner(
    int32_t max_query_length, int32_t max_target_length,
    int32_t max_alignments, AlignmentType type,
    DefaultDeviceAllocator allocator, cudaStream_t stream, int32_t device_id)
{
    if (type == AlignmentType::global_alignment)
    {
        return std::make_unique<AlignerGlobalHirschbergMyers>(max_query_length, max_target_length, max_alignments, allocator, stream, device_id);
    }
    else
    {
        throw std::runtime_error("Aligner for specified type not implemented yet.");
    }
}

std::unique_ptr<Aligner> create_aligner(
    int32_t max_query_length, int32_t max_target_length,
    int32_t max_alignments, AlignmentType type,
    cudaStream_t stream, int32_t device_id, int64_t max_device_memory_allocator_caching_size)
{
    if (max_device_memory_allocator_caching_size < -1)
    {
        throw std::invalid_argument("max_device_memory_allocator_caching_size has to be either -1 (=all available GPU memory) or greater or equal than 0.");
    }
#ifdef GW_ENABLE_CACHING_ALLOCATOR
    // uses CachingDeviceAllocator
    if (max_device_memory_allocator_caching_size == -1)
    {
        max_device_memory_allocator_caching_size = genomeworks::cudautils::find_largest_contiguous_device_memory_section();
        if (max_device_memory_allocator_caching_size == 0)
        {
            throw std::runtime_error("No memory available for caching");
        }
    }
    genomeworks::DefaultDeviceAllocator allocator(max_device_memory_allocator_caching_size);
#else
    // uses CudaMallocAllocator
    genomeworks::DefaultDeviceAllocator allocator;
#endif
    return create_aligner(max_query_length, max_target_length, max_alignments, type, allocator, stream, device_id);
}
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
