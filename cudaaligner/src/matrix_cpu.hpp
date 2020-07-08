/*
* Copyright 2019-2020 NVIDIA CORPORATION.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include <vector>
#include <cassert>
#include <iostream>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

template <typename T>
class matrix
{
public:
    using value_type = T;
    static inline bool error(int t)
    {
        printf("assert: %d", t);
        return false;
    }

    matrix()              = default;
    matrix(matrix const&) = default;
    matrix(matrix&&)      = default;
    matrix& operator=(matrix const&) = default;
    matrix& operator=(matrix&&) = default;
    ~matrix()                   = default;

    matrix(int n, int m, T value = 0)
        : data_(n * m, value)
        , n(n)
        , m(m)
    {
    }

    inline T const& operator()(int i, int j) const
    {
        assert(0 <= i || error(i));
        assert(i < n || error(i));
        assert(0 <= j || error(j));
        assert(j < m || error(j));
        return data_[i + n * j];
    }

    inline T& operator()(int i, int j)
    {
        assert(0 <= i || error(i));
        assert(i < n || error(i));
        assert(0 <= j || error(j));
        assert(j < m || error(j));
        return data_[i + n * j];
    }

    inline T* data()
    {
        return data_.data();
    }

    inline void print(std::ostream& o) const
    {
        for (int i = 0; i < n; ++i)
        {
            o << "\n";
            for (int j = 0; j < m; ++j)
                o << (*this)(i, j) << "\t";
        }
        o << std::endl;
    }

    inline int num_rows() const
    {
        return n;
    }
    inline int num_cols() const
    {
        return m;
    }

private:
    std::vector<T> data_;
    int n = 0;
    int m = 0;
};

template <typename T>
inline bool operator==(matrix<T> const& a, matrix<T> const& b)
{
    if (a.num_rows() != b.num_rows() || a.num_cols() != b.num_cols())
        return false;
    int n = a.num_rows();
    int m = a.num_cols();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
        {
            if (a(i, j) != b(i, j))
                return false;
        }
    return true;
}

template <typename T>
inline bool operator!=(matrix<T> const& a, matrix<T> const& b)
{
    return !(a == b);
}
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
