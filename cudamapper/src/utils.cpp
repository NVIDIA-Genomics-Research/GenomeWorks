#include "utils.hpp"
#include <limits>

namespace genomeworks {

    static std::uint64_t kmer_to_representation(const std::string& basepairs, std::size_t start_element, std::size_t length) {
        std::uint64_t minimizer = 0;
        if (length <= 2*sizeof(std::uint64_t)) { // two basepairs per byte due to 4-bit packing
            // TODO: Lexical ordering for now, this will change in the future
            std::uint64_t a = 0b000;
            std::uint64_t c = 0b001;
            std::uint64_t g = 0b010;
            std::uint64_t t = 0b011;
            std::uint64_t x = 0b100;
            for (std::size_t i = start_element; i < start_element + length; ++i) {
                minimizer <<= 4;
                switch(basepairs[i]) {
                    case 'A': minimizer |= a; break;
                    case 'C': minimizer |= c; break;
                    case 'G': minimizer |= g; break;
                    case 'T': minimizer |= t; break;
                    default : minimizer |= x; break;
                }
            }
        } else {
            // TODO: throw?
            minimizer = std::numeric_limits<std::uint64_t>::max();
        }
        return minimizer;
    }

}

