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

#include <claraparabricks/genomeworks/logging/logging.hpp>

#include <iostream>
#include <fstream>
#include <memory>
#include <cassert>

namespace claraparabricks
{

namespace genomeworks
{

namespace logging
{
static std::unique_ptr<std::ostream> out_stream_ = nullptr;

static LogLevel level_ = LogLevel::error;

void check_logger()
{
    if (out_stream_ == nullptr)
    {
        std::cerr << "GenomeWorks logger not initialized yet. Initializing default logger now." << std::endl;
        initialize_logger(LogLevel::error);
    }
}

std::string log_level_str(LogLevel level)
{
    std::string prefix;
    switch (level)
    {
    case critical: prefix = "CRITICAL"; break;
    case error: prefix = "ERROR"; break;
    case warn: prefix = "WARN"; break;
    case info: prefix = "INFO"; break;
    case debug: prefix = "DEBUG"; break;
    default:
        assert(false); // Unknown log level
        prefix = "INFO";
        break;
    }
    return prefix;
}

void initialize_logger(LogLevel level, const char* filename)
{
    if (out_stream_ == nullptr)
    {
        level_ = level;
        if (filename == nullptr)
        {
            out_stream_ = std::make_unique<std::ofstream>(filename);
        }
        else
        {
            out_stream_ = std::make_unique<std::ostream>(std::cerr.rdbuf());
        }
        *out_stream_ << "Initialized GenomeWorks logger with log level " << log_level_str(level_) << std::endl;
    }
    else
    {
        *out_stream_ << "Logger already initialized with log level " << log_level_str(level_) << std::endl;
    }
}
void log(LogLevel level, const char* file, int32_t line, const char* msg)
{
    check_logger();
    if (level <= level_)
    {
        *out_stream_ << "[" << log_level_str(level) << " " << file << ":" << line << "] " << msg << std::endl;
    }
}

} // namespace logging

} // namespace genomeworks

} // namespace claraparabricks
