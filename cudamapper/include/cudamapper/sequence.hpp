#pragma once

#include <memory>
#include <string>

namespace genomeworks {
    /// Sequence - represents an individual read (name, DNA nucleotides etc)
    class Sequence {
    public:
        /// Sequence name
        virtual const std::string &name() const = 0;

        /// Sequence data. Typically this is a sequence of bases A,C,T,G
        virtual const std::string &data() const = 0;

        /// create_sequence - return a Sequence object
        /// \return Sequence implementation
        static std::unique_ptr<Sequence> create_sequence();
    };
}
