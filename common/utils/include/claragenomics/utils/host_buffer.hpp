#pragma once
#include <claragenomics/utils/buffer.hpp>

namespace claragenomics
{

template <typename T>
using host_buffer = buffer<T, hostAllocator>;

} // namespace claragenomics
