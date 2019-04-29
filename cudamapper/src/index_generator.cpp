#include <string>
#include <iostream>
#include "bioparser/bioparser.hpp"
#include "cudamapper/sequence.hpp"
#include "index_generator_cpu.hpp"

namespace genomeworks {
    std::unique_ptr<IndexGenerator> IndexGenerator::create_index_generator(std::uint64_t minimizer_size, std::uint64_t window_size) {
        return std::make_unique<IndexGeneratorCPU>(minimizer_size, window_size);
    }
}
