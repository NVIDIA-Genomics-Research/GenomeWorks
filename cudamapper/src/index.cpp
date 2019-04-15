#include <string>
#include <iostream>
#include "bioparser/bioparser.hpp"
#include "cudamapper/sequence.hpp"
#include "cpu_index.hpp"

namespace genomeworks {
    std::unique_ptr<Index> Index::create_index() {
        return std::make_unique<CPUIndex>();
    }
}