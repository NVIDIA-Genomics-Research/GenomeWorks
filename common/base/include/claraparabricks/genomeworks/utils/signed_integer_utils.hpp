

#pragma once

#include <limits>
#include <cassert>
#include <stdexcept>
#include <type_traits>

namespace claraparabricks
{

namespace genomeworks
{

template <class Container>
typename std::make_signed<typename Container::size_type>::type get_size(Container const& c)
{
    using size_type   = typename Container::size_type;
    using signed_type = typename std::make_signed<size_type>::type;
    assert(c.size() <= static_cast<size_type>(std::numeric_limits<signed_type>::max()));
    return static_cast<signed_type>(c.size());
}

template <class Integer, class Container>
Integer get_size(Container const& c)
{
    using signed_type = Integer;
    assert(c.size() <= static_cast<typename Container::size_type>(std::numeric_limits<signed_type>::max()));
    return static_cast<signed_type>(c.size());
}

template <class T>
T throw_on_negative(T x, const char* message)
{
    static_assert(std::is_arithmetic<T>::value, "throw_on_negative expects an arithmetic type.");
    if (x < T(0))
        throw std::invalid_argument(message);
    return x;
}

} // namespace genomeworks

} // namespace claraparabricks
