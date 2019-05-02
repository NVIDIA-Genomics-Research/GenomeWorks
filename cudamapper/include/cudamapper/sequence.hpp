#pragma once

#include <memory>
#include <string>

namespace genomeworks {
/// \addtogroup cudamapper
/// \{

    /// Sequence - represents an individual read (name, DNA nucleotides etc)
    class Sequence {
    public:
        /// Sequence name
        virtual const std::string &name() const = 0;

        /// Sequence data. Typically this is a sequence of bases A,C,T,G
        virtual const std::string &data() const = 0;

        /// create_sequence - return a Sequence object
        /// \return Sequence implementation
        static std::unique_ptr<Sequence> create_sequence(const char *name, uint32_t name_length, const char *data,
                                                         uint32_t data_length);

        /// create_sequence - return a Sequence object
        /// \return Sequence implementation
        static std::unique_ptr<Sequence> create_sequence(const std::string &name, const std::string &data);
    };

/// \}
}
