#include <string>
#include <iostream>
#include "bioparser/bioparser.hpp"
#include "cudamapper/sequence.hpp"
#include "cpu_index.hpp"

namespace genomeworks {
    std::unique_ptr<Index> Index::create_index(std::uint64_t minimizer_size, std::uint64_t window_size) {
        return std::make_unique<CPUIndex>(minimizer_size, window_size);
    }
}
