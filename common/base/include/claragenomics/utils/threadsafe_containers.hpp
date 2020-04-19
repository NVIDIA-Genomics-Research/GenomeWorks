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

#include <atomic>
#include <functional>
#include <vector>

#include <claragenomics/types.hpp>

namespace claragenomics
{

/// ThreadsafeDataProvider - wrapper around std::vector which gives elements one by one to multiple threads and signals when there are no elements left
template <typename T>
class ThreadsafeDataProvider
{
public:
    /// @brief default constructor
    ThreadsafeDataProvider()
        : data_()
        , counter_(0)
    {
    }

    /// @brief Constructor
    /// @param data
    ThreadsafeDataProvider(const std::vector<T>& data)
        : data_(data)
        , counter_(0)
    {
    }

    /// @brief Constructor
    /// @param data
    ThreadsafeDataProvider(std::vector<T>&& data)
        : data_(data)
        , counter_(0)
    {
    }

    /// @brief deleted copy constructor
    /// @param rhs
    ThreadsafeDataProvider(const ThreadsafeDataProvider& rhs) = delete;

    /// @brief deleted copy assignment operator
    /// @param rhs
    ThreadsafeDataProvider& operator==(const ThreadsafeDataProvider& rhs) = delete;

    /// @brief deleted move constructor
    /// @param rhs
    ThreadsafeDataProvider(ThreadsafeDataProvider&& rhs) = delete;

    /// @brief deleted move assignment operator
    /// @param rhs
    ThreadsafeDataProvider& operator==(ThreadsafeDataProvider&& rhs) = delete;

    /// @brief destructor
    ~ThreadsafeDataProvider() = default;

    /// \brief returns next available element or empty optional object if there are no elements left
    cga_optional_t<T> next_element()
    {
        size_t my_counter = counter_++;
#ifdef __cpp_lib_optional
        return my_counter < data_.size() ? cga_optional_t<T>(std::move(data_[my_counter])) : std::nullopt;
#else
        return my_counter < data_.size() ? cga_optional_t<T>(std::move(data_[my_counter])) : std::experimental::nullopt;
#endif
    }

private:
    /// data to provide, whenever an element is provided it is moved from this value
    std::vector<T> data_;
    /// number of elements provided so far
    std::atomic<std::size_t> counter_;
};

} // namespace claragenomics
