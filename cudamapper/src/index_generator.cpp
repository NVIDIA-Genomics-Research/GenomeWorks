#include "bioparser/bioparser.hpp"
#include "index_generator_cpu.hpp"

namespace genomeworks {
    std::unique_ptr<IndexGenerator> IndexGenerator::create_index_generator(const std::string &query_filename, std::uint64_t minimizer_size, std::uint64_t window_size) {
        return std::make_unique<IndexGeneratorCPU>(query_filename, minimizer_size, window_size);
    }
}
