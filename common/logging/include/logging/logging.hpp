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

/// \ingroup logging
/// \{

#define gw_log_level_debug 0
#define gw_log_level_info 1
#define gw_log_level_warn 2
#define gw_log_level_error 3
#define gw_log_level_critical 4
#define gw_log_level_off 5

#ifndef GW_LOG_LEVEL
#ifndef NDEBUG
#define GW_LOG_LEVEL gw_log_level_debug
#else // NDEBUG
#define GW_LOG_LEVEL gw_log_level_error
#endif // NDEBUG
#endif // GW_LOG_LEVEL

#if GW_LOG_LEVEL == gw_log_level_info
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO
#elif GW_LOG_LEVEL == gw_log_level_debug
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#elif GW_LOG_LEVEL == gw_log_level_warn
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_WARN
#elif GW_LOG_LEVEL == gw_log_level_error
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_ERROR
#elif GW_LOG_LEVEL == gw_log_level_critical
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_CRITICAL
#else
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF
#endif

// MUST come after the defines of the logging level!
#include <spdlog/spdlog.h>

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
#define GW_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)

/// \ingroup logging
/// \def GW_LOG_INFO
/// \brief Log at info level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)

/// \ingroup logging
/// \def GW_LOG_WARN
/// \brief Log at warning level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)

/// \ingroup logging
/// \def GW_LOG_ERROR
/// \brief Log at error level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)

/// \ingroup logging
/// \def GW_LOG_CRITICAL
/// \brief Log at fatal/critical error level (does NOT exit)
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)

} // namespace logging
} // namespace genomeworks

/// \}
