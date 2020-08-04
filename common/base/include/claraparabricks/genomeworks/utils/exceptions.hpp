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

#include <exception>

namespace claraparabricks
{

namespace genomeworks
{

/// @brief Exception class for out-of-(device-)memory errors.
///
/// Exceptions of this class are thrown if a memory allocation fails on the device.
class device_memory_allocation_exception : public std::exception
{
public:
    device_memory_allocation_exception() = default;
    /// Copy constructor
    device_memory_allocation_exception(device_memory_allocation_exception const&) = default;
    /// Move constructor
    device_memory_allocation_exception(device_memory_allocation_exception&&) = default;
    /// Assignment
    device_memory_allocation_exception& operator=(device_memory_allocation_exception const&) = default;
    /// Move-Assignment
    device_memory_allocation_exception& operator=(device_memory_allocation_exception&&) = default;
    /// Destructor
    virtual ~device_memory_allocation_exception() = default;

    /// Returns the error message of the exception
    virtual const char* what() const noexcept
    {
        return "Could not allocate device memory!";
    }
};

} // namespace genomeworks

} // namespace claraparabricks
