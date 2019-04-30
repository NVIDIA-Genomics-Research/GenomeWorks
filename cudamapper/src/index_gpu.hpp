#pragma once

#include "cudamapper/index.hpp"
#include "index_generator_cpu.hpp"

namespace genomeworks {
    /// IndexGPU - index of minimizers
    ///
    /// Index of (k,w)-minimizers suitable for GPU processing
    class IndexGPU : public Index {
    public:
        /// \brief Constructs the index using index_generator
        ///
        /// \param index_generator
        IndexGPU(IndexGenerator& index_generator);

        /// \brief constructs an emptry index
        IndexGPU();
    };
}
