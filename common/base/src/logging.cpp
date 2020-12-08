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

namespace claraparabricks
{

namespace genomeworks
{

namespace logging
{
static std::unique_ptr<std::ostream> out_stream_ = nullptr;
static std::unique_ptr<std::ofstream> out_file_  = nullptr;

static LogLevel level_ = LogLevel::ERROR;

void check_logger()
{
    if (out_stream_ == nullptr)
    {
        std::cerr << "GenomeWorks logger not initialized yet. Initializing default logger now." << std::endl;
        create_logger(LogLevel::ERROR);
    }
}

std::string log_level_str(LogLevel level)
{
    std::string prefix;
    switch (level)
    {
    case CRITICAL: prefix = "CRITICAL"; break;
    case ERROR: prefix = "ERROR"; break;
    case WARN: prefix = "WARN"; break;
    case INFO: prefix = "INFO"; break;
    case DEBUG: prefix = "DEBUG"; break;
    default: throw std::runtime_error("Unknown Log Level passed.\n");
    }
    return prefix;
}

void create_logger(LogLevel level, const std::string& filename)
{
    if (out_stream_ == nullptr)
    {
        std::streambuf* buffer = nullptr;
        level_                 = level;
        if (filename != "")
        {
            out_file_ = std::make_unique<std::ofstream>(filename);
            buffer    = out_file_->rdbuf();
        }
        else
        {
            buffer = std::cerr.rdbuf();
        }
        out_stream_ = std::make_unique<std::ostream>(buffer);
        *out_stream_ << "Initialized GenomeWorks logger with log level " << log_level_str(level_) << std::endl;
    }
    else
    {
        *out_stream_ << "Logger already initialized with log level " << log_level_str(level_) << std::endl;
    }
}
void log(LogLevel level, const std::string& file, int32_t line, const std::string& msg)
{
    check_logger();
    if (level <= level_)
    {
        std::string prefix = log_level_str(level);
        *out_stream_ << "[" << prefix << " " << file << ":" << line << "] " << msg << std::endl;
    }
}

} // namespace logging

} // namespace genomeworks

} // namespace claraparabricks
