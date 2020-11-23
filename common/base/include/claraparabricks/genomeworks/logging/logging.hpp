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
/// \file
/// \defgroup logging Internal logging package
/// Base docs for the logging package
/// This package makes use of SpdLog under the following license:
///
/// The MIT License (MIT)
///
/// Copyright (c) 2016 Gabi Melman.
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

#include <claraparabricks/genomeworks/gw_config.hpp>

/// \ingroup logging
/// \{

/// \brief DEBUG log level
#define gw_log_level_debug 0
/// \brief INFO log level
#define gw_log_level_info 1
/// \brief WARN log level
#define gw_log_level_warn 2
/// \brief ERROR log level
#define gw_log_level_error 3
/// \brief CRITICAL log level
#define gw_log_level_critical 4
/// \brief No logging
#define gw_log_level_off 5

#ifndef GW_LOG_LEVEL
#ifndef NDEBUG
/// \brief Defines the logging level used in the current module
#define GW_LOG_LEVEL gw_log_level_debug
#else // NDEBUG
/// \brief Defines the logging level used in the current module
#define GW_LOG_LEVEL gw_log_level_error
#endif // NDEBUG
#endif // GW_LOG_LEVEL

#if GW_LOG_LEVEL == gw_log_level_info
/// \brief Set log level to INFO
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#elif GW_LOG_LEVEL == gw_log_level_debug
/// \brief Set log level to DEBUG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#elif GW_LOG_LEVEL == gw_log_level_warn
/// \brief Set log level to WARN
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
#elif GW_LOG_LEVEL == gw_log_level_error
/// \brief Set log level to ERROR
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_ERROR
#elif GW_LOG_LEVEL == gw_log_level_critical
/// \brief Set log level to CRITICAL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_CRITICAL
#else
/// \brief Set log level to OFF
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF
#endif

// MUST come after the defines of the logging level!
#ifdef GW_CUDA_BEFORE_9_2
// Due to a header file incompatibility with nvcc in CUDA 9.0
// logging through the logger class in GW is disabled for any .cu files.
#pragma message("Logging disabled for CUDA Toolkit < 9.2")
#elif __GNUC__ >= 9
// Due to a ISO C++ standard incompatibility the spdlog fails to pass
// pedantic requirements.
#pragma message("Logging disabled for GCC >= 9")
#else
#include <spdlog/spdlog.h>
#endif

namespace claraparabricks
{

namespace genomeworks
{

namespace logging
{
/// \ingroup logging
/// Logging status type
enum class LoggingStatus
{
    success = 0,       ///< Success
    cannot_open_file,  ///< Initialization could not open the output file requested
    cannot_open_stdout ///< Stdout could not be opened for logging
};

/// \ingroup logging
/// Init Initialize the logging
/// \param filename if specified, the path/name of the file into which logging should be placed.
/// The default is stdout
/// \return success or error status
LoggingStatus Init(const char* filename = nullptr);

/// \ingroup logging
/// SetHeader Adjust the header/preface for each log message
/// \param logTime if true, the detailed time will be prepended to each message.
/// \param logLocation if true, the file and line location logging will be prepended to each message.
/// \return success or error status
LoggingStatus SetHeader(bool logTime, bool logLocation);

/// \ingroup logging
/// \def GW_LOG_DEBUG
/// \brief Log at debug level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#ifdef GW_CUDA_BEFORE_9_2
#define GW_LOG_DEBUG(...)
#elif __GNUC__ >= 9
#define GW_LOG_DEBUG(...)
#else
#define GW_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#endif

/// \ingroup logging
/// \def GW_LOG_INFO
/// \brief Log at info level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#ifdef GW_CUDA_BEFORE_9_2
#define GW_LOG_INFO(...)
#elif __GNUC__ >= 9
#define GW_LOG_INFO(...)
#else
#define GW_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#endif

/// \ingroup logging
/// \def GW_LOG_WARN
/// \brief Log at warning level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#ifdef GW_CUDA_BEFORE_9_2
#define GW_LOG_WARN(...)
#elif __GNUC__ >= 9
#define GW_LOG_WARN(...)
#else
#define GW_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#endif

/// \ingroup logging
/// \def GW_LOG_ERROR
/// \brief Log at error level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#ifdef GW_CUDA_BEFORE_9_2
#define GW_LOG_ERROR(...)
#elif __GNUC__ >= 9
#define GW_LOG_ERROR(...)
#else
#define GW_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#endif

/// \ingroup logging
/// \def GW_LOG_CRITICAL
/// \brief Log at fatal/critical error level (does NOT exit)
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#ifdef GW_CUDA_BEFORE_9_2
#define GW_LOG_CRITICAL(...)
#elif __GNUC__ >= 9
#define GW_LOG_CRITICAL(...)
#else
#define GW_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)
#endif

} // namespace logging

} // namespace genomeworks

} // namespace claraparabricks

/// \}
