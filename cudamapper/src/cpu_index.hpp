#pragma once

#include "cudamapper/index.hpp"

namespace genomeworks {
    /// CPUIndex - generates and manages (k,w)-minimizer index for one or more sequences
    /// lifecycle managed by the host (not GPU)
    class CPUIndex : public Index {
    public:
        /// \brief generate an in-memory (k,w)-minimizer index
        /// \param query_filename
        void generate_index(std::string query_filename);
    };
}
