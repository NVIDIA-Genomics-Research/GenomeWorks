

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
// or higher, thegw_optional_t and gw_string_view_t aliases
// can be removed and std::optional and std::string_view can
// be used directly instead
#if __cplusplus >= 201703
template <typename T>
using gw_optional_t               = std::optional<T>;
using gw_nullopt_t                = std::nullopt_t;
constexpr gw_nullopt_t gw_nullopt = std::nullopt;
using gw_string_view_t            = std::string_view;
#else
template <typename T>
using gw_optional_t               = std::experimental::optional<T>;
using gw_nullopt_t                = std::experimental::nullopt_t;
constexpr gw_nullopt_t gw_nullopt = std::experimental::nullopt;
using gw_string_view_t            = std::experimental::string_view;
#endif

} // namespace genomeworks

} // namespace claraparabricks
