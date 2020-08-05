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

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

namespace claraparabricks
{

namespace genomeworks
{

namespace logging
{
static std::shared_ptr<spdlog::logger> logger = nullptr;

LoggingStatus Init(const char* filename)
{
    // for now, first call wins:
    if (logger != nullptr)
        return LoggingStatus::success;

    if (filename != nullptr)
    {
        try
        {
            logger = spdlog::basic_logger_mt("GWLogger", filename);
        }
        catch (const spdlog::spdlog_ex& ex)
        {
            return LoggingStatus::cannot_open_file;
        }
    }
    else
    {
        try
        {
            logger = spdlog::stderr_logger_mt("GWLogger");
        }
        catch (const spdlog::spdlog_ex& ex)
        {
            return LoggingStatus::cannot_open_stdout;
        }
    }

    spdlog::set_default_logger(logger);

#ifdef _DEBUG
    SetHeader(true, true);
#else
    SetHeader(false, false);
#endif

    spdlog::flush_every(std::chrono::seconds(1));

    return LoggingStatus::success;
}

LoggingStatus SetHeader(bool logTime, bool logLocation)
{
    std::string pattern = "";

    if (logTime)
        pattern = pattern + "[%H:%M:%S %z]";

    if (logLocation)
        pattern = pattern + "[%@]";

    pattern = pattern + "%v";

    spdlog::set_pattern(pattern);

    return LoggingStatus::success;
}
} // namespace logging

} // namespace genomeworks

} // namespace claraparabricks
