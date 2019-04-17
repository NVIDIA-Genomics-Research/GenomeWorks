#include "minimizer.hpp"

namespace genomeworks {

    Minimizer::Minimizer(std::uint64_t representation, std::size_t position, std::uint64_t sequence_id)
    : representation_(representation), position_(position), sequence_id_(sequence_id)
    {}

    std::uint64_t Minimizer::representation() const { return representation_; }

    std::size_t Minimizer::position() const { return position_; }

    std::size_t Minimizer::sequence_id() const { return sequence_id_; }

}

