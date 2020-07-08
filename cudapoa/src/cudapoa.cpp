

#include <claraparabricks/genomeworks/cudapoa/cudapoa.hpp>
#include <claraparabricks/genomeworks/logging/logging.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudapoa
{

StatusType Init()
{
    if (logging::LoggingStatus::success != logging::Init())
        return StatusType::generic_error;

    return StatusType::success;
}

} // namespace cudapoa

} // namespace genomeworks

} // namespace claraparabricks
