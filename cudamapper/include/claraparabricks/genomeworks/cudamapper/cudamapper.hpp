

#pragma once

/// \defgroup cudamapper CUDA mapper package
/// Base docs for the cudamapper package (tbd)
/// \{

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{
enum class StatusType
{
    success = 0,
    generic_error
};

StatusType Init();
}; // namespace cudamapper
}; // namespace genomeworks

} // namespace claraparabricks

/// \}
