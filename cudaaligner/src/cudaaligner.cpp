

#include <claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp>
#include <claraparabricks/genomeworks/logging/logging.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudaaligner
{

StatusType Init()
{
    if (logging::LoggingStatus::success != logging::Init())
        return StatusType::generic_error;

    return StatusType::success;
}
} // namespace cudaaligner

} // namespace genomeworks

} // namespace claraparabricks
