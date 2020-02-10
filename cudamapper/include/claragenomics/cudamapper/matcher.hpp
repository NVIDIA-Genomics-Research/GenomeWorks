/*
* Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <memory>
#include <thrust/device_vector.h>
#include <claragenomics/cudamapper/index.hpp>
#include <claragenomics/utils/device_buffer.hpp>

namespace claragenomics
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
    /// \return matcher
    static std::unique_ptr<Matcher> create_matcher(std::shared_ptr<DeviceAllocator> allocator,
                                                   const Index& query_index,
                                                   const Index& target_index);
};

/// \}

} // namespace cudamapper

} // namespace claragenomics
