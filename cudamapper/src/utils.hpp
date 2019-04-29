#pragma once

#include <cstdint>
#include <string>

namespace genomeworks {

    /// \brief Is the representation value of forward or reverse compliment sequence
    enum class RepresentationDirection {
        FORWARD,
        REVERSE
    };

    /// \brief representation and its direction
    struct KmerIntegerRepresentation {
        std::uint64_t representation_;
        RepresentationDirection direction_;
    };

    /// \brief Converts a kmer of length length into 4-bit packed numeric representation
    ///
    /// Representation uses lexicographical ordering. It returns the smaller of forward and reverse complement representation
    ///
    /// \param baseparis
    /// \param start_element where in basepairs the kmer actually starts 
    /// \param length lenght of the kmer
    ///
    /// \return representation and direction of the sequence
    KmerIntegerRepresentation kmer_to_integer_representation(const std::string& basepairs, std::size_t start_element, std::size_t length);

}

