#pragma once

#include <cstdlib>

namespace genomeworks {

namespace cudaaligner {

inline std::string generate_random_genome(const uint32_t length)
{
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    std::string genome = "";
    for(uint32_t i = 0; i < length; i++)
    {
        genome += alphabet[rand() % 4];
    }
    return genome;
}

} // cudaaligner

} // genomeworks
