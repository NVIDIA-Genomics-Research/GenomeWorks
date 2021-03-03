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
    static inline bool error(int32_t t)
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

    matrix(int32_t n, int32_t m, T value = 0)
        : data_(n * m, value)
        , n_rows_(n)
        , n_cols_(m)
    {
    }

    inline T const& operator()(int32_t i, int32_t j) const
    {
        assert(0 <= i || error(i));
        assert(i < n_rows_ || error(i));
        assert(0 <= j || error(j));
        assert(j < n_cols_ || error(j));
        return data_[i + n_rows_ * j];
    }

    inline T& operator()(int32_t i, int32_t j)
    {
        assert(0 <= i || error(i));
        assert(i < n_rows_ || error(i));
        assert(0 <= j || error(j));
        assert(j < n_cols_ || error(j));
        return data_[i + n_rows_ * j];
    }

    inline T* data()
    {
        return data_.data();
    }

    inline void print(std::ostream& o) const
    {
        for (int32_t i = 0; i < n_rows_; ++i)
        {
            o << "\n";
            for (int32_t j = 0; j < n_cols_; ++j)
                o << (*this)(i, j) << "\t";
        }
        o << std::endl;
    }

    inline int32_t num_rows() const
    {
        return n_rows_;
    }
    inline int32_t num_cols() const
    {
        return n_cols_;
    }

private:
    std::vector<T> data_;
    int32_t n_rows_ = 0;
    int32_t n_cols_ = 0;
};

template <typename T>
inline bool operator==(matrix<T> const& a, matrix<T> const& b)
{
    if (a.num_rows() != b.num_rows() || a.num_cols() != b.num_cols())
        return false;
    const int32_t n = a.num_rows();
    const int32_t m = a.num_cols();
    for (int32_t i = 0; i < n; ++i)
        for (int32_t j = 0; j < m; ++j)
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
