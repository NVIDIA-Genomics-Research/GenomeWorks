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

#include <cstdint>
#include <string>

/// \ingroup logging
/// \{

namespace claraparabricks
{

namespace genomeworks
{

namespace logging
{

/// GenomeWorks Logging levels.
enum LogLevel
{
    CRITICAL = 0,
    ERROR,
    WARN,
    INFO,
    DEBUG
};

/// Initialize logger across GenomeWorks.
/// \param [in] level LogLevel for logger.
/// \param [in] filename File to redirect log messages to.
void create_logger(LogLevel level, const std::string& filename = "");

/// Log messages to logger.
/// \param [in] level LogLevel for message.
/// \param [in] file Filename for originating message.
/// \param [in] line Line number for originating message.
/// \param [in] msg Content of log message.
void log(LogLevel level, const std::string& file, int32_t line, const std::string& msg);

/// \ingroup logging
/// \def GW_LOG_DEBUG
/// \brief Log at debug level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_DEBUG(msg) claraparabricks::genomeworks::logging::log(claraparabricks::genomeworks::logging::LogLevel::DEBUG, __FILE__, __LINE__, msg)

/// \ingroup logging
/// \def GW_LOG_INFO
/// \brief Log at info level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_INFO(msg) claraparabricks::genomeworks::logging::log(claraparabricks::genomeworks::logging::LogLevel::INFO, __FILE__, __LINE__, msg)

/// \ingroup logging
/// \def GW_LOG_WARN
/// \brief Log at warning level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_WARN(msg) claraparabricks::genomeworks::logging::log(claraparabricks::genomeworks::logging::LogLevel::WARN, __FILE__, __LINE__, msg)

/// \ingroup logging
/// \def GW_LOG_ERROR
/// \brief Log at error level
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_ERROR(msg) claraparabricks::genomeworks::logging::log(claraparabricks::genomeworks::logging::LogLevel::ERROR, __FILE__, __LINE__, msg)

/// \ingroup logging
/// \def GW_LOG_CRITICAL
/// \brief Log at fatal/critical error level (does NOT exit)
///
/// parameters as per https://github.com/gabime/spdlog/blob/v1.x/README.md
#define GW_LOG_CRITICAL(msg) claraparabricks::genomeworks::logging::log(claraparabricks::genomeworks::logging::LogLevel::CRITICAL, __FILE__, __LINE__, msg)
} // namespace logging

} // namespace genomeworks

} // namespace claraparabricks

/// \}
