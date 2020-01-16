#pragma once
#include <claragenomics/utils/buffer.hpp>

namespace claragenomics
{

template <typename T>
using device_buffer = buffer<T, deviceAllocator>;

template <typename T, typename Allocator>
void swap(buffer<T, Allocator>& a,
          buffer<T, Allocator>& b)
{
    a.swap(b);
}

} // namespace claragenomics
