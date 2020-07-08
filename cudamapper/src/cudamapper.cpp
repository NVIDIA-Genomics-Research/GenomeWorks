

#include <claraparabricks/genomeworks/cudamapper/cudamapper.hpp>
#include <claraparabricks/genomeworks/logging/logging.hpp>

namespace claraparabricks
{

namespace genomeworks
{

namespace cudamapper
{

namespace cudamapper
{

StatusType Init()
{
    if (logging::LoggingStatus::success != logging::Init())
        return StatusType::generic_error;

    return StatusType::success;
}

}; // namespace cudamapper
} // namespace cudamapper

} // namespace genomeworks

} // namespace claraparabricks
