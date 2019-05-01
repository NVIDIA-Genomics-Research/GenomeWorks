#include <cudapoa/cudapoa.hpp>
#include <logging/logging.hpp>

namespace genomeworks {
    namespace cudapoa {

        StatusType Init()
        {
            if (logging::LoggingStatus::success != logging::Init())
                return StatusType::generic_error;
                
            return StatusType::success;
        }

    }
}
