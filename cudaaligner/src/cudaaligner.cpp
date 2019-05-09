#include "cudaaligner/cudaaligner.hpp"
#include <logging/logging.hpp>

namespace genomeworks {

namespace cudaaligner {

StatusType Init()
{
    if (logging::LoggingStatus::success != logging::Init())
        return StatusType::generic_error;

    return StatusType::success;
}

}

}
