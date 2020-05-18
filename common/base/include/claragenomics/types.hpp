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

#include <cstdint>

#if __cplusplus >= 201703
#include <optional>
#include <string_view>
#else
#include <experimental/optional>
#include <experimental/string_view>
#endif

namespace claraparabricks
{

namespace genomeworks
{

/// unique ID of a read
using read_id_t = std::uint32_t;

/// number of reads
using number_of_reads_t = read_id_t;

/// position of a basepair/kmer in a read
using position_in_read_t = std::uint32_t;

/// number of basepairs
using number_of_basepairs_t = position_in_read_t;

// TODO: Once minimal supported GCC version is moved to GCC 7.1
// or higher, thecga_optional_t and cga_string_view_t aliases 
// can be removed and std::optional and std::string_view can
// be used directly instead
#if __cplusplus >= 201703
template <typename T>
using cga_optional_t                = std::optional<T>;
using cga_nullopt_t                 = std::nullopt_t;
constexpr cga_nullopt_t cga_nullopt = std::nullopt;
using cga_string_view_t             = std::string_view;
#else
template <typename T>
using cga_optional_t                = std::experimental::optional<T>;
using cga_nullopt_t                 = std::experimental::nullopt_t;
constexpr cga_nullopt_t cga_nullopt = std::experimental::nullopt;
using cga_string_view_t             = std::experimental::string_view;
#endif

} // namespace genomeworks

} // namespace claraparabricks
