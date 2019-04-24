#pragma once

#include <cstdint>

namespace genomeworks {

    /// Minimizer - represents one occurrance of a minimizer
    class Minimizer {
    public:
        /// \brief constructor
        ///
        /// \param representation 4-bit packed representation of a kmer
        /// \param position position of the minimizer in the sequence
        /// \param sequence_id sequence's id
        Minimizer(std::uint64_t representation, std::size_t position, std::uint64_t sequence_id);

        /// \brief returns minimizers representation
        /// \return minimizer representation
        std::uint64_t representation() const;

        /// \brief returns position of the minimizer in the sequence
        /// \return position of the minimizer in the sequence
        std::size_t position() const;

        /// \brief returns sequence's ID
        /// \return sequence's ID
        std::size_t sequence_id() const;

    private:
        std::uint64_t representation_; // supports up to 2*64 basepairs in a minimzer. Normaly minimizers of around 20 elements are used
        std::size_t position_;
        std::size_t sequence_id_;
    };

}

