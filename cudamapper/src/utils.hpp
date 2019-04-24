#pragma once

#include <cstdint>
#include <string>

namespace genomeworks {

    /// \brief Converts a kmer of length length into 4-bit packed numeric representation
    ///
    /// \param baseparis
    /// \param start_element where in basepairs the kmer actually starts 
    /// \param length lenght of the kmer
    static std::uint64_t kmer_to_representation(const std::string& basepairs, std::size_t start_element, std::size_t length);

}

