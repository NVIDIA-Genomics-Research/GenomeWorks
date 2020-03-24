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

#include <exception>

namespace claragenomics
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

} // namespace claragenomics
