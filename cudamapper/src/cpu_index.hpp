#pragma once

#include "cudamapper/index.hpp"

namespace genomeworks {
    class CPUIndex : public Index {
    public:
        void generate_index(std::string query_filename);
    };
}
