#include <cudapoa/cudapoa.hpp>
#include <logging/logging.hpp>

namespace genomeworks {
    namespace cudapoa {

        StatusType Init()
        {
            logging::Init();
            return StatusType::SUCCESS;
        }

    }
}
