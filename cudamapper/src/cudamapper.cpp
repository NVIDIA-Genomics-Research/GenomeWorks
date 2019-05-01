#include <cudamapper/cudamapper.hpp>
#include <logging/logging.hpp>

namespace genomeworks {
    namespace cudamapper {

        StatusType Init()
        {
            if (logging::LoggingStatus::success != logging::Init())
                return StatusType::generic_error;
                
            return StatusType::success;
        }

    };
};
