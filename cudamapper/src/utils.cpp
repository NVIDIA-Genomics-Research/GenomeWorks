#include "utils.hpp"
#include <algorithm>
#include <limits>

namespace genomeworks {

    static std::uint64_t kmer_to_integer_representation(const std::string& basepairs, std::size_t start_element, std::size_t length) {
        std::uint64_t forward_representation = 0;
        std::uint64_t reverse_representation = 0;
        if (length <= 2*sizeof(std::uint64_t)) { // two basepairs per byte due to 4-bit packing
            // TODO: Lexical ordering for now, this will change in the future
            std::uint64_t a = 0b000;
            std::uint64_t c = 0b001;
            std::uint64_t g = 0b010;
            std::uint64_t t = 0b011;
            std::uint64_t x = 0b100;
            // in reverse complement base 0 goes to position length-1, 1 to length-2...
            for (std::size_t i = start_element, position_reverse = length - 1; i < start_element + length; ++i, --position_reverse) {
                forward_representation <<= 4;
                switch(basepairs[i]) {
                    case 'A': forward_representation |= a; reverse_representation |= t << 4*position_reverse; break;
                    case 'C': forward_representation |= c; reverse_representation |= g << 4*position_reverse; break;
                    case 'G': forward_representation |= g; reverse_representation |= c << 4*position_reverse; break;
                    case 'T': forward_representation |= t; reverse_representation |= a << 4*position_reverse; break;
                    default : forward_representation |= x; reverse_representation |= x << 4*position_reverse; break;
                }
            }
        } else {
            // TODO: throw?
            forward_representation = reverse_representation = std::numeric_limits<std::uint64_t>::max();
        }
        return std::min(forward_representation, reverse_representation);
    }

}

