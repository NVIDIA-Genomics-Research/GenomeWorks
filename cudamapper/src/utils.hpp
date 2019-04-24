#pragma once

#include <cstdint>
#include <string>

namespace genomeworks {

    /// \brief Converts a k-mer of length length into 4-bit packed numeric representation
    ///
    /// \param baseparis
    /// \param start_element where in basepairs the k-mer actually starts 
    /// \param length lenght of the k-mer
    static std::uint64_t k_mer_to_representation(const std::string& basepairs, std::size_t start_element, std::size_t length);

}

