

#pragma once

#include <memory>
#include <claraparabricks/genomeworks/cudamapper/index.hpp>
#include <claraparabricks/genomeworks/utils/device_buffer.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
/// \addtogroup cudamapper
/// \{

/// Matcher - base matcher
class Matcher
{
public:
    /// \brief Virtual destructor
    virtual ~Matcher() = default;

    /// \brief returns anchors
    /// \return anchors
    virtual device_buffer<Anchor>& anchors() = 0;

    /// \brief Creates a Matcher object
    /// \param allocator The device memory allocator to use for buffer allocations
    /// \param query_index
    /// \param target_index
    /// \param cuda_stream CUDA stream on which the work is to be done. Device arrays are also associated with this stream and will not be freed at least until all work issued on this stream before calling their destructor is done
    /// \return matcher
    static std::unique_ptr<Matcher> create_matcher(DefaultDeviceAllocator allocator,
                                                   const Index& query_index,
                                                   const Index& target_index,
                                                   const cudaStream_t cuda_stream = 0);
};

/// \}

} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
