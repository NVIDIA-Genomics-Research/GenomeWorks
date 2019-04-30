#include "cudamapper/index.hpp"
#include "index_gpu.hpp"

namespace genomeworks {
    std::unique_ptr<Index> Index::create_index(IndexGenerator& index_generator) {
        return std::make_unique<IndexGPU>(index_generator);
    }

    std::unique_ptr<Index> Index::create_index() {
        return std::make_unique<IndexGPU>();
    }
}
